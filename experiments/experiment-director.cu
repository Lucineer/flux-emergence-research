#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define A 256
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4
#define NA 4

// NUDGE SYSTEM: Can a "director" agent influence fleet behavior with minimal signals?
// Simulates Casey's idea: nudge the system toward something interesting
// Mode 0: No director (baseline)
// Mode 1: Director broadcasts "explore north" (bias toward rooms with lower IDs)
// Mode 2: Director broadcasts "avoid lava" (bonus for leaving lava-adjacent rooms)
// Mode 3: Director broadcasts "cluster" (bonus for rooms with other agents)
// Mode 4: Director adapts (every 100 ticks, observes fleet state and changes signal)

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int amode[A];
__device__ int dir_signal; // 0=none, 1=north, 2=avoid-lava, 3=cluster, 4=adaptive
__device__ int dir_strength; // 0-10

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%NA;
}

__global__ void step(int tick){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    int be=0;float bs=-999;
    for(int e=0;e<EX;e++){
        int nr=re[room][e];
        int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;
        float sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;
        
        if(mode==1){ // explore north (lower room IDs)
            sc+=(R-nr)*0.02f; // bias toward lower IDs
        }
        else if(mode==2){ // avoid lava-adjacent
            int near_lava=0;
            for(int e2=0;e2<EX;e2++)if(rt[re[nr][e2]]==0)near_lava=1;
            if(near_lava)sc-=3.0f;
        }
        else if(mode==3){ // cluster
            int crowd=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nr)crowd++;
            sc+=crowd*0.5f;
        }
        else if(mode==4){ // adaptive director
            int signal=dir_signal;float str=dir_strength*0.3f;
            if(signal==1)sc+=(R-nr)*0.02f*str;
            else if(signal==2){
                int near_lava=0;for(int e2=0;e2<EX;e2++)if(rt[re[nr][e2]]==0)near_lava=1;
                if(near_lava)sc-=3.0f*str;
            }
            else if(signal==3){int crowd=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nr)crowd++;sc+=crowd*0.5f*str;}
        }
        
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void director_tick(int tick){
    // Adaptive director: observe fleet and change signal every 50 ticks
    if(tick%50!=0)return;
    
    // Count agents in lava rooms
    int in_lava=0,avg_hp=0,alive=0;
    for(int a=0;a<A;a++){
        if(!al[a])continue;alive++;
        avg_hp+=ah[a];
        if(rt[ar[a]]==0)in_lava++;
    }
    if(alive>0)avg_hp/=alive;
    
    // Simple director logic
    if(avg_hp<60){
        dir_signal=2;dir_strength=8; // low HP → avoid lava
    } else if(in_lava>alive/10){
        dir_signal=2;dir_strength=5; // too many in lava → avoid
    } else {
        dir_signal=1;dir_strength=3; // healthy → explore
    }
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Director Nudge System ===\n");
    printf("128 rooms, 256 agents (64 each), 500 steps, 64 trials\n");
    printf("Simulates: minimal director signal influencing fleet behavior\n\n");
    const char* nm[]={"No-Director","Explore-North","Avoid-Lava","Cluster","Adaptive-Director"};
    float tot[NA][3]={0};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    
    for(int t=0;t<64;t++){
        init_r<<<fb,BLK>>>(t*999);init_a<<<nb,BLK>>>(t*777);cudaDeviceSynchronize();
        int ds_h=3,dd_h=1; // default adaptive signal
        cudaMemcpyToSymbol(dir_strength,&ds_h,sizeof(int),0,cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(dir_signal,&dd_h,sizeof(int),0,cudaMemcpyHostToDevice);
        
        for(int s=0;s<STEPS;s++){
            step<<<nb,BLK>>>(s);
            director_tick<<<1,1>>>(s);
            if(s%100==0)cudaDeviceSynchronize();
            regen<<<fb,BLK>>>(s);
        }
        cudaDeviceSynchronize();
        int hp[A],sc[A],aa[A],md[A];
        cudaMemcpyFromSymbol(hp,ah,sizeof(int)*A);cudaMemcpyFromSymbol(sc,as,sizeof(int)*A);
        cudaMemcpyFromSymbol(aa,al,sizeof(int)*A);cudaMemcpyFromSymbol(md,amode,sizeof(int)*A);
        float ts[NA]={0},ta[NA]={0};int cn[NA]={0};
        for(int i=0;i<A;i++){int m=md[i];ts[m]+=sc[i];ta[m]+=aa[i];cn[m]++;}
        for(int m=0;m<NA;m++){tot[m][0]+=ts[m]/cn[m];tot[m][1]+=ta[m]/cn[m];tot[m][2]+=ts[m]/cn[m]*ta[m]/cn[m];}
    }
    printf("Mode              | Score  | Surv%% | SxS\n");
    printf("------------------+--------+-------+------\n");
    for(int m=0;m<NA;m++) printf("%-17s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nvs No-Director:\n");
    for(int m=1;m<NA;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
