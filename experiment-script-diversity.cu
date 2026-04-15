#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NDIVERSITY 4
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

// diversity_mode: 0=uniform, 1=random angles, 2=two families, 3=gradient
__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w,
    int diversity_mode, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    
    switch(diversity_mode){
    case 0: // Uniform: golden angle spread
        for(int i=0;i<8;i++) script_dir[i]=base_angle+i*0.785f;
        break;
    case 1: // Random: each agent has random script
        for(int i=0;i<8;i++) script_dir[i]=cr(&rng)*6.2832f;
        break;
    case 2: // Two families: CW vs CCW
        { float dir = (tid < n/2) ? 1.0f : -1.0f;
          for(int i=0;i<8;i++) script_dir[i]=base_angle+i*0.785f*dir; }
        break;
    case 3: // Gradient: speed varies by agent ID
        for(int i=0;i<8;i++) script_dir[i]=base_angle+i*0.785f;
        break;
    }
    
    float agent_speed = 2.0f;
    if(diversity_mode == 3) agent_speed = 1.0f + (float)(tid % 16) * 0.2f; // 1.0 to 4.0 gradient
    
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;
        float dx=cosf(script_dir[p])*agent_speed;
        float dy=sinf(script_dir[p])*agent_speed;
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
    const char* modes[]={"Uniform","Random","TwoFam","Gradient"};
    printf("=== SCRIPT DIVERSITY: Law 208 ===\n");
    printf("N=%d Food=%d Steps=%d Speed=1x Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    printf("%-10s | %-8s | %-8s | %-8s\n","Mode","Score","Surv","FleetTot");
    printf("-----------|----------|----------|----------\n");
    
    float best_total=0;int best_mode=0;
    float mode_totals[NDIVERSITY];
    
    for(int m=0;m<NDIVERSITY;m++){
        float ts=0,ta=0,tt=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,1.0f,STEPS,N,FOOD,W,m,(unsigned int)(42+tr*1111+m*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];
            cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;float total=0;
            for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];total+=hs[i];}
            ts+=avg/N;ta+=ac;tt+=total;
        }
        ts/=TRIALS;float surv=ta/TRIALS/N*100;tt/=TRIALS;
        mode_totals[m] = tt;
        if(tt>best_total){best_total=tt;best_mode=m;}
        printf("%-10s | %6.3f   | %5.1f%% | %6.1f\n",modes[m],ts,surv,tt);
    }
    printf("\n=== ANALYSIS ===\n");
    printf("Best fleet total: %s (%.1f total food collected)\n",modes[best_mode],best_total);
    
    // Find baseline (Uniform mode) total for comparison
    float baseline_total = mode_totals[0];
    printf("Baseline (Uniform): %.1f total food\n", baseline_total);
    
    printf(">> Law 208: Script diversity %s fleet total food collection\n",
        best_total > baseline_total ? "increases" : "does not change");
    
    // Show percentage change
    if (best_total > baseline_total) {
        float percent_increase = ((best_total - baseline_total) / baseline_total) * 100.0f;
        printf("Improvement: +%.1f%% over uniform script\n", percent_increase);
    }
    
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}