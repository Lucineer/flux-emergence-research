#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define A 256
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4
#define NA 5

// DIRECTOR AS PERCEPTION ENHANCER (not behavior nudger)
// Instead of "go north", give agents better sensors
// Mode 0: No director, basic perception
// Mode 1: Director provides "1-hop danger preview" (can see neighbors' terrain)
// Mode 2: Director provides "resource density heatmap" (aggregate room scores)
// Mode 3: Director provides "death prediction" (rooms where agents are likely to die)
// Mode 4: Director provides ALL enhanced perception

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int amode[A];
__device__ int room_aggregate[R]; // director-computed room value scores
__device__ int death_predicted[R]; // director-predicted danger

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

// Director kernel: computes aggregate room scores (expensive, runs infrequently)
__global__ void director_compute(){
    for(int i=0;i<R;i++){
        int s=0;
        if(rt[i]==0)s-=10;else if(rt[i]==2)s+=5;
        for(int j=0;j<IT;j++)s+=ri[i][j];
        s+=rg[i]/5;
        // Add neighbor danger info
        int danger=0;for(int e=0;e<EX;e++)if(rt[re[i][e]]==0)danger++;
        death_predicted[i]=(danger>=2)?1:0;
        room_aggregate[i]=s;
    }
}

__global__ void step(){
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
        
        if(mode==1){ // 1-hop danger preview
            for(int e2=0;e2<EX;e2++){if(rt[re[nr][e2]]==0)sc-=1.0f;}
        }
        else if(mode==2){ // resource density heatmap
            sc+=room_aggregate[nr]*0.02f;
        }
        else if(mode==3){ // death prediction
            if(death_predicted[nr])sc-=3.0f;
        }
        else if(mode==4){ // ALL perception
            for(int e2=0;e2<EX;e2++){if(rt[re[nr][e2]]==0)sc-=1.0f;}
            sc+=room_aggregate[nr]*0.02f;
            if(death_predicted[nr])sc-=3.0f;
        }
        
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Director as Perception Enhancer ===\n");
    printf("128 rooms, 256 agents (52 each), 500 steps, 64 trials\n\n");
    const char* nm[]={"No-Director","Danger-Preview","Res-Heatmap","Death-Predict","All-Perception"};
    float tot[NA][3]={0};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    for(int t=0;t<64;t++){
        init_r<<<fb,BLK>>>(t*999);init_a<<<nb,BLK>>>(t*777);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){
            if(s%10==0)director_compute<<<1,1>>>();
            step<<<nb,BLK>>>();if(s%100==0)cudaDeviceSynchronize();regen<<<fb,BLK>>>(s);
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
