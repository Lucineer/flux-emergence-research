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

// DEATH OBSERVATION: Can agents learn from watching others die?
// Mode 0: No observation (baseline)
// Mode 1: Observe deaths in current room, avoid rooms where deaths happened
// Mode 2: Observe deaths AND correlate with terrain, learn terrain-danger mapping
// Mode 3: Born with death-avoidance (hardcoded: avoid rooms where >2 deaths seen)

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int amode[A];
__device__ int death_count[R]; // deaths observed per room
__device__ int death_map[A][R]; // per-agent learned death map
__device__ int total_deaths;

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
    death_count[i]=0;
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%NA;
    for(int r=0;r<R;r++)death_map[i][r]=0;
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    // Observe deaths (from previous tick) and update death maps
    if(mode>=1&&mode<=2){
        int deaths_here=death_count[room];
        if(deaths_here>0){
            death_map[i][room]+=deaths_here;
            // Mode 2: also learn about neighbor rooms (terrain correlation)
            if(mode==2){
                for(int e=0;e<EX;e++){
                    int nr=re[room][e];
                    // If room is lava, neighbors might also be dangerous
                    if(rt[room]==0)death_map[i][nr]+=1;
                }
            }
        }
    }
    if(mode==3){
        // Born with avoidance: check global death count
        if(death_count[room]>2){
            ah[i]-=5; // panic
        }
    }

    int be=0;float bs=-999;
    for(int e=0;e<EX;e++){
        int nr=re[room][e];float sc=0;
        {int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
        
        // Death avoidance bias
        if(mode==1) sc-=death_map[i][nr]*0.5f;
        else if(mode==2) sc-=death_map[i][nr]*0.3f;
        else if(mode==3&&death_count[nr]>2) sc-=5.0f;
        
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0){
        al[i]=0;atomicAdd(&death_count[room],1);atomicAdd(&total_deaths,1);}
}

__global__ void regen(int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}
    if(s%100==0)death_count[i]=0; // reset death observations periodically
}

int main(){
    printf("=== Death Observation Learning ===\n");
    printf("128 rooms, 256 agents (64 each), 500 steps, 64 trials\n\n");
    const char* nm[]={"No-Obs","Observe-Deaths","Terrain-Corr","Born-Avoid"};
    float tot[NA][3]={0};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    int td_h[NA]={0}; // total deaths per mode
    
    for(int t=0;t<64;t++){
        int zero=0;cudaMemcpyToSymbol(total_deaths,&zero,sizeof(int),0,cudaMemcpyHostToDevice);
        init_r<<<fb,BLK>>>(t*999);init_a<<<nb,BLK>>>(t*777);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){step<<<nb,BLK>>>();if(s%100==0)cudaDeviceSynchronize();regen<<<fb,BLK>>>(s);}
        cudaDeviceSynchronize();
        int hp[A],sc[A],aa[A],md[A];
        cudaMemcpyFromSymbol(hp,ah,sizeof(int)*A);cudaMemcpyFromSymbol(sc,as,sizeof(int)*A);
        cudaMemcpyFromSymbol(aa,al,sizeof(int)*A);cudaMemcpyFromSymbol(md,amode,sizeof(int)*A);
        float ts[NA]={0},ta[NA]={0};int cn[NA]={0};
        for(int i=0;i<A;i++){int m=md[i];ts[m]+=sc[i];ta[m]+=aa[i];cn[m]++;}
        for(int m=0;m<NA;m++){tot[m][0]+=ts[m]/cn[m];tot[m][1]+=ta[m]/cn[m];tot[m][2]+=ts[m]/cn[m]*ta[m]/cn[m];}
    }
    
    printf("Mode          | Score  | Surv%% | SxS\n");
    printf("--------------+--------+-------+------\n");
    for(int m=0;m<NA;m++) printf("%-13s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nvs No-Observation:\n");
    for(int m=1;m<NA;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
