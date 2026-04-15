#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSIZES 5
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    for (int i=0;i<8;i++) script_dir[i] = base_angle + i*0.785f;
    for (int t=0;t<steps&&energy>0;t++){
        int p=t%8;
        float dx=cosf(script_dir[p])*2.0f, dy=sinf(script_dir[p])*2.0f;
        float dist=sqrtf(dx*dx+dy*dy);
        energy-=0.005f+dist*0.003f;
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
    int sizes[NSIZES]={128,256,512,1024,2048};
    printf("=== SCRIPT DENSITY AT HIGH SPEED: Law 204 ===\n");
    printf("Food=%d Steps=%d Speed=16x World=%d Trials=%d\n\n",FOOD,STEPS,W,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,2048*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,2048*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    float res[NSIZES], srv[NSIZES];
    for(int s=0;s<NSIZES;s++){
        int n=sizes[s]; float ts=0,ta=0;
        int blk=(n+BLK-1)/BLK;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,16.0f,STEPS,n,FOOD,W,(unsigned int)(42+tr*1111+s*111));
            cudaDeviceSynchronize();
            float hs[2048]; int ha[2048];
            cudaMemcpy(hs,d_s,n*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,n*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;for(int i=0;i<n;i++){avg+=hs[i];ac+=ha[i];}
            ts+=avg/n;ta+=ac;
        }
        res[s]=ts/TRIALS; srv[s]=ta/TRIALS/n*100;
        printf("%d agents: score=%.3f survival=%.1f%%\n",n,res[s],srv[s]);
    }
    printf("\n=== ANALYSIS ===\n");
    if(res[0]>0&&res[NSIZES-1]>0){
        float ratio = res[NSIZES-1]/res[0];
        printf("Per-agent score ratio (2048/128): %.2fx\n",ratio);
        if(res[NSIZES-1]>res[0]) printf(">> Law 204: Fleet effect applies to scripted agents — more agents = higher per-agent fitness\n");
        else printf(">> Fleet effect does NOT apply to scripted agents at high speed\n");
    }
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}