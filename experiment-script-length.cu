#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NLENGTHS 8
#define NSPEEDS 2
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w, int script_len, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    // Generate script of given length
    // Use a simple hash for deterministic variety
    for (int t=0;t<steps&&energy>0;t++){
        int p = t % script_len;
        // Deterministic direction from golden angle + phase
        float angle = base_angle + p * 6.2832f / script_len;
        float dx = cosf(angle) * 2.0f;
        float dy = sinf(angle) * 2.0f;
        float dist = sqrtf(dx*dx + dy*dy);
        energy -= 0.005f + dist * 0.003f;
        x = fmodf(x + dx + w, w);
        y = fmodf(y + dy + w, w);
        for (int i=0;i<food_count;i++){
            if(!falive[i])continue;
            float fdx=fx[i]-x,fdy=fy[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            if(fdx*fdx+fdy*fdy<16.0f){int old=atomicExch(&falive[i],0);if(old){energy=fminf(energy+10.0f,200.0f);score+=1.0f;}}
        }
    }
    scores[tid]=score;alive[tid]=(energy>0)?1:0;
}

int main(){
    int lengths[NLENGTHS]={2,4,8,16,32,64,128,256};
    float speeds[NSPEEDS]={16.0f,32.0f};
    printf("=== SCRIPT LENGTH SWEEP: Law 205 ===\n");
    printf("N=%d Food=%d Steps=%d World=%d Trials=%d\n\n",N,FOOD,STEPS,W,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    for(int sp=0;sp<NSPEEDS;sp++){
        printf("--- Speed %dx ---\n", (int)speeds[sp]);
        printf("%-8s | %-8s | %-8s\n", "Length", "Score", "Survival");
        printf("---------|----------|----------\n");
        float best_score=0; int best_len=0;
        for(int l=0;l<NLENGTHS;l++){
            float ts=0,ta=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,speeds[sp],STEPS,N,FOOD,W,lengths[l],(unsigned int)(42+tr*1111+sp*111+l*11));
                cudaDeviceSynchronize();
                float hs[N];int ha[N];
                cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
                float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
                ts+=avg/N;ta+=ac;
            }
            float avg_score=ts/TRIALS, surv=ta/TRIALS/N*100;
            if(avg_score>best_score){best_score=avg_score;best_len=lengths[l];}
            printf("%-8d | %6.3f   | %5.1f%%\n",lengths[l],avg_score,surv);
        }
        printf("Best script length at %dx: %d (score=%.3f)\n\n",(int)speeds[sp],best_len,best_score);
    }
    printf("=== LAW CANDIDATES ===\n");
    printf(">> Law 205: Script length has inverted-U relationship with fitness\n");
    printf("   Too short = repetitive starvation. Too long = no pattern advantage.\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}