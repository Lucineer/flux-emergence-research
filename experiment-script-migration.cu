#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSPEEDS 5
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, float *fx, float *fy, int *falive,
    float food_speed, int use_drift_bias, int steps, int n, int food_count, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    float drift_dir=0.0f; // food drifts in +x direction
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        if(use_drift_bias){dx+=cosf(drift_dir)*0.5f;dy+=sinf(drift_dir)*0.5f;}
        float dist=sqrtf(dx*dx+dy*dy);energy-=0.005f+dist*0.003f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        for(int i=0;i<food_count;i++){
            if(!falive[i])continue;
            float fdx=fx[i]-x,fdy=fy[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            if(fdx*fdx+fdy*fdy<16.0f){int old=atomicExch(&falive[i],0);if(old){energy=fminf(energy+10.0f,200.0f);score+=1.0f;}}
        }
    }
    scores[tid]=score;
}

__global__ void move_food(float *fx, float *fy, int *falive, int food_count, int w, float speed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count || !falive[idx]) return;
    fx[idx] = fmodf(fx[idx] + speed + w, w);
}

int main(){
    float speeds[NSPEEDS]={0.0f,0.5f,1.0f,2.0f,4.0f};
    const char* sn[]={"0.0","0.5","1.0","2.0","4.0"};
    printf("=== SCRIPT MIGRATION: Law 239 ===\nN=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    int blk=(N+BLK-1)/BLK;int fblk=(FOOD+BLK-1)/BLK;
    printf("%-8s | %-8s | %-8s | %-8s\n","Drift","Pure","Bias","Lift");
    printf("--------|----------|----------|----------\n");
    for(int sp=0;sp<NSPEEDS;sp++){
        float tp=0,tb=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            for(int chunk=0;chunk<STEPS;chunk+=100){
                simulate<<<blk,BLK>>>(d_s,d_fx,d_fy,d_fa,speeds[sp],0,100,N,FOOD,W,(unsigned int)(42+tr*1111+sp*111+chunk*11));
                move_food<<<fblk,BLK>>>(d_fx,d_fy,d_fa,FOOD,W,speeds[sp]);
            }
            cudaDeviceSynchronize();
            float hs[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            float avg=0;for(int i=0;i<N;i++)avg+=hs[i];tp+=avg/N;
            // Bias
            cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            for(int chunk=0;chunk<STEPS;chunk+=100){
                simulate<<<blk,BLK>>>(d_s,d_fx,d_fy,d_fa,speeds[sp],1,100,N,FOOD,W,(unsigned int)(42+tr*1111+sp*111+chunk*11));
                move_food<<<fblk,BLK>>>(d_fx,d_fy,d_fa,FOOD,W,speeds[sp]);
            }
            cudaDeviceSynchronize();
            cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            avg=0;for(int i=0;i<N;i++)avg+=hs[i];tb+=avg/N;
        }
        tp/=TRIALS;tb/=TRIALS;float lift=(tp>0)?(tb-tp)/tp*100:0;
        printf("%-8s | %6.3f   | %6.3f   | %+.1f%%\n",sn[sp],tp,tb,lift);
    }
    printf("\n>> Law 239: Does directional bias help scripts follow food migration?\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);
    return 0;
}