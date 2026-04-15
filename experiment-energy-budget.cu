#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NENERGY 8
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w,
    float start_energy, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = start_energy, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    for(int i=0;i<8;i++) script_dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;
        float dx=cosf(script_dir[p])*2.0f,dy=sinf(script_dir[p])*2.0f;
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
    float energies[NENERGY]={20,40,60,80,100,120,150,200};
    printf("=== ENERGY BUDGET AT 32x: Law 210 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    printf("%-8s | %-8s | %-8s | %-10s\n","Energy","Score","Surv","Food/Agent");
    printf("--------|----------|----------|------------\n");
    
    float threshold=0;
    for(int e=0;e<NENERGY;e++){
        float ts=0,ta=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,energies[e],(unsigned int)(42+tr*1111+e*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];
            cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
            ts+=avg/N;ta+=ac;
        }
        float avg_score=ts/TRIALS,surv=ta/TRIALS/N*100;
        printf("%-8.0f | %6.3f   | %5.1f%% | %8.3f\n",energies[e],avg_score,surv,avg_score);
        if(surv>=90.0f && threshold==0) threshold=energies[e];
    }
    
    // Calculate energy cost per step
    float cost_per_step = 0.005f + 2.0f * 0.003f; // base + dist*0.003 for speed=2
    float min_energy = cost_per_step * STEPS;
    printf("\n=== ANALYSIS ===\n");
    printf("Energy cost per step (scripted, speed=2): %.4f\n",cost_per_step);
    printf("Minimum energy for %d steps (no food): %.1f\n",STEPS,min_energy);
    if(threshold>0) printf(">> Law 210: Minimum starting energy for scripted survival: ~%.0f\n",threshold);
    printf("   Each food item provides +10 energy (capped at 200).\n");
    printf("   Agents need to find food within first ~%.0f steps or starve.\n",
        (energies[0]-min_energy)/(-cost_per_step));
    
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}