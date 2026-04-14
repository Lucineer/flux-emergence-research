#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define A 256
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4
#define NA 10

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int aprev_r[A],aprev_h[A],aprev_g[A];
__device__ int r_agent_d[R],r_gold_d[R],r_item_d[R],r_death_d[R];
__device__ float r_hp_d[R];
__device__ float r_ema[R][4]; // 4 EMA channels per room
__device__ float a_lat[A][4],a_w[A][16];
__device__ float alpha_g; // EMA alpha

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
    r_agent_d[i]=0;r_gold_d[i]=0;r_item_d[i]=0;r_hp_d[i]=0;r_death_d[i]=0;
    for(int j=0;j<4;j++)r_ema[i][j]=0;
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    aprev_r[i]=ar[i];aprev_h[i]=100;aprev_g[i]=10;
    for(int r=0;r<4;r++)a_lat[i][r]=0;
    for(int r=0;r<4;r++)for(int c=0;c<4;c++)
        a_w[i][r*4+c]=(r==c)?0.8f+cr(&aseed[i])%40/100.0f:cr(&aseed[i])%20/100.0f-0.1f;
}

__global__ void clear_d(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    r_agent_d[i]=0;r_gold_d[i]=0;r_item_d[i]=0;r_hp_d[i]=0;r_death_d[i]=0;}

__global__ void rec_d(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int c=ar[i],p=aprev_r[i];
    if(c!=p){atomicAdd(&r_agent_d[c],1);atomicAdd(&r_agent_d[p],-1);}
    int gc=ag[i]-aprev_g[i];if(gc>0)atomicAdd(&r_gold_d[ar[i]],gc);
    int hl=aprev_h[i]-ah[i];if(hl>0)atomicAdd(&r_hp_d[ar[i]],hl);
}

__global__ void update_ema(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    r_ema[i][0]=alpha_g*(float)r_agent_d[i]+(1-alpha_g)*r_ema[i][0];
    r_ema[i][1]=alpha_g*(float)r_gold_d[i]/20.0f+(1-alpha_g)*r_ema[i][1];
    r_ema[i][2]=alpha_g*(float)r_item_d[i]/4.0f+(1-alpha_g)*r_ema[i][2];
    r_ema[i][3]=alpha_g*(float)r_death_d[i]+(1-alpha_g)*r_ema[i][3];
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aprev_r[i]=ar[i];aprev_h[i]=ah[i];aprev_g[i]=ag[i];
    if(!al[i])return;
    int room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    int be=0;float bs=-999;float pred[4];
    for(int e=0;e<EX;e++){
        int nr=re[room][e];
        float o0=fminf(fmaxf(r_ema[nr][0],-1.0f),1.0f);
        float o1=fminf(fmaxf(r_ema[nr][1],-1.0f),1.0f);
        float o2=fminf(r_ema[nr][2],1.0f);
        float o3=fminf(r_ema[nr][3],1.0f);
        float sc=o1*3.0f+o2*2.0f-o3*8.0f;
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    // Learn
    float act[4]={r_ema[nw][0],r_ema[nw][1],r_ema[nw][2],r_ema[nw][3]};
    for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=a_w[i][r*4+c]*a_lat[i][c];pred[r]=fminf(1.0f,fmaxf(-1.0f,pred[r]));}
    float lr=0.02f;
    for(int r=0;r<4;r++){float e=act[r]-pred[r];for(int c=0;c<4;c++)a_w[i][r*4+c]+=lr*e*a_lat[i][c];}
    for(int r=0;r<4;r++)a_lat[i][r]=act[r];

    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0){al[i]=0;atomicAdd(&r_death_d[nw],1);}
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== EMA Window Sweep for Delta Perception ===\n");
    printf("256 agents, 128 rooms, 500 steps, 32 trials\n\n");
    printf("Alpha | Window~ | Score  | Surv%% | SxS\n");
    printf("------+--------+--------+-------+------\n");

    float alphas[]={1.0,0.5,0.33,0.2,0.1,0.05,0.02,0.01,0.005};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;

    // Hardcoded baseline first
    // (reuse same kernel but skip JEPA — just run hardcoded heuristic)
    float base_sc=0,base_sv=0;
    // Actually let's just compare EMA modes against each other

    for(int ai=0;ai<NA;ai++){
        float al=alphas[ai];
        cudaMemcpyToSymbol(alpha_g,&al,sizeof(float));
        float tsc=0,tsv=0;
        for(int t=0;t<32;t++){
            init_r<<<fb,BLK>>>(t*999);init_a<<<nb,BLK>>>(t*777);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){clear_d<<<fb,BLK>>>();
                step<<<nb,BLK>>>();cudaDeviceSynchronize();rec_d<<<nb,BLK>>>();
                update_ema<<<fb,BLK>>>();regen<<<fb,BLK>>>(s);if(s%100==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hp[A],sc[A],aa[A];
            cudaMemcpyFromSymbol(hp,ah,sizeof(int)*A);cudaMemcpyFromSymbol(sc,as,sizeof(int)*A);
            cudaMemcpyFromSymbol(aa,al,sizeof(int)*A);
            for(int i=0;i<A;i++){tsc+=sc[i];tsv+=aa[i];}
        }
        tsc/=32*A;tsv/=32*A;
        float window=1.0f/al;
        printf(" %.3f | %6.0f  | %6.1f | %5.1f | %.0f\n",al,window,tsc,tsv*100,tsc*tsv*100);
    }
    return 0;
}
