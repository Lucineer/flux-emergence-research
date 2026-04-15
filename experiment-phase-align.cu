#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NPHASES 4
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, int phase_mode, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];
    for(int i=0;i<8;i++) dir[i]=base_angle+i*0.785f;
    int offset = 0;
    if(phase_mode==0) offset=0;
    else if(phase_mode==1) offset=tid%8;
    else if(phase_mode==2) offset=(int)(base_angle/0.785f)%8;
    else offset=(tid<n/2)?0:4;
    for(int t=0;t<steps&&energy>0;t++){
        int p=(t+offset)%8;
        float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
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
    scores[tid]=score;alive[tid]=(energy>0)?1:0;
}

int main(){
    const char* nm[]={"InSync","AgentID","GoldenAngle","AntiPhase"};
    printf("=== PHASE ALIGNMENT: Law 219 ===\nN=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    float results[NPHASES];
    for(int p=0;p<NPHASES;p++){
        float ts=0,ta=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,p,(unsigned int)(42+tr*1111+p*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
            ts+=avg/N;ta+=ac;
        }
        results[p]=ts/TRIALS;
        printf("%-14s: score=%.3f surv=%.1f%%\n",nm[p],results[p],ta/TRIALS/N*100);
    }
    float best=results[0];int bp=0;
    for(int p=1;p<NPHASES;p++)if(results[p]>best){best=results[p];bp=p;}
    printf("\nBest: %s (%.3f)\n",nm[bp],best);
    if(results[0]<best) printf(">> Law 219: Phase offset helps by %.1f%%\n",(best-results[0])/results[0]*100);
    else printf(">> Law 219: Phase offset does NOT help — diversity must be in direction not timing\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}