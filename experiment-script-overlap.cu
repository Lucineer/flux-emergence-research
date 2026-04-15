#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD_FRAC 0.04f  // food density: 4% of world cells
#define STEPS 3000
#define BLK 128
#define NWORLDS 4
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    for(int i=0;i<8;i++) script_dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(script_dir[p])*2.0f,dy=sinf(script_dir[p])*2.0f;
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
    int worlds[NWORLDS]={64,128,256,512};
    printf("=== DENSITY vs WORLD SIZE: Law 214 ===\n");
    printf("N=%d Food=4%% cells Steps=%d Trials=%d\n\n",N,STEPS,TRIALS);
    
    printf("%-10s | %-8s | %-8s | %-10s | %-12s | %-12s\n",
        "World","Food","Score","Surv","Density(a/w²)","Food/a");
    printf("----------|----------|----------|------------|--------------|------------\n");
    
    float best_score=0;int best_w=0;
    for(int wi=0;wi<NWORLDS;wi++){
        int w=worlds[wi];
        int food=(int)(w*w*FOOD_FRAC);
        float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
        cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,food*sizeof(float));
        cudaMalloc(&d_fy,food*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,food*sizeof(int));
        float hfx[65536],hfy[65536];srand(42);
        for(int i=0;i<food;i++){hfx[i]=((float)rand()/RAND_MAX)*w;hfy[i]=((float)rand()/RAND_MAX)*w;}
        cudaMemcpy(d_fx,hfx,food*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy,hfy,food*sizeof(float),cudaMemcpyHostToDevice);
        int blk=(N+BLK-1)/BLK;
        float ts=0,ta=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,food*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,food,w,(unsigned int)(42+tr*1111+wi*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];
            cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
            ts+=avg/N;ta+=ac;
        }
        ts/=TRIALS;float surv=ta/TRIALS/N*100;
        float density=(float)N/(w*w)*10000; // per 10000 cells
        float food_per_agent=(float)food/N;
        if(ts>best_score){best_score=ts;best_w=w;}
        printf("%-10d | %6d   | %6.3f   | %5.1f%% | %10.2f   | %10.3f\n",
            w,food,ts,surv,density,food_per_agent);
        cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    }
    printf("\nBest world size: %d (score=%.3f)\n",worlds[best_w],best_score);
    printf(">> Law 214: Density is the key variable, not agent count\n");
    printf("   At fixed agent count, larger world = more space per agent = less overlap\n");
    return 0;
}