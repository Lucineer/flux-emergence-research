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

// HYBRID: Static + Delta. Can we get the score gain without the survival loss?
// Mode 0: Static only
// Mode 1: Static + Delta bonus (use static as base, add delta signal)
// Mode 2: Static + Delta penalty (use static base, subtract if delta negative)
// Mode 3: Static primary, Delta tiebreaker (use delta only when scores are close)
// Mode 4: Adaptive: use static when HP low, delta when HP high

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int amode[A];
__device__ int prev_score[A];

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
    prev_score[i]=0;
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    int be=0;float bs=-999;
    float scores[EX];
    for(int e=0;e<EX;e++){
        int nr=re[room][e];
        int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;
        float sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;
        scores[e]=sc;
        
        if(mode==1) sc+=(sc-prev_score[i])*0.3f; // delta bonus
        else if(mode==2){
            float delta=sc-prev_score[i];
            if(delta<0)sc+=delta*0.5f; // penalize going to worse rooms
        }
        else if(mode==3){
            // Static primary, delta as tiebreaker (handled after loop)
        }
        else if(mode==4){
            // Adaptive: HP>60 use delta, HP<=60 use static
            float delta=sc-prev_score[i];
            if(ah[i]>60) sc=delta*1.5f;
        }
        
        if(sc>bs){bs=sc;be=e;}
    }
    
    // Mode 3 tiebreaker
    if(mode==3){
        float best_st=scores[be];int best2=-1;float bs2=-999;
        for(int e=0;e<EX;e++){
            if(e==be)continue;
            float diff=abs(scores[e]-best_st);
            if(diff<2.0f&&scores[e]>bs2){bs2=scores[e];best2=e;}
        }
        if(best2>=0){
            float d1=scores[be]-prev_score[i];
            float d2=scores[best2]-prev_score[i];
            if(d2>d1)be=best2; // tiebreak: prefer improving
        }
    }
    
    int nw=re[room][be];
    int r=0;for(int j=0;j<IT;j++)if(ri[nw][j]>0)r++;
    prev_score[i]=r*2.0f+rg[nw]*0.1f-rt[nw]*0.5f;
    ar[i]=nw;
    
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Hybrid Static+Delta Navigation ===\n");
    printf("128 rooms, 256 agents (52 each), 500 steps, 64 trials\n\n");
    const char* nm[]={"Static","Static+Delta","Static-DeltaPen","Static+Tiebreak","Adaptive-HP"};
    float tot[NA][3]={0};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    for(int t=0;t<64;t++){
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
    printf("Mode              | Score  | Surv%% | SxS\n");
    printf("------------------+--------+-------+------\n");
    for(int m=0;m<NA;m++) printf("%-17s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nvs Static:\n");
    for(int m=1;m<NA;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
