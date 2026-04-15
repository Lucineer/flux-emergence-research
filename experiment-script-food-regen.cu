#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSIZES 4
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, int regen_rate, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
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

__global__ void regen_food(float *fx, float *fy, int *falive, int food_count, int w,
    int rate, int step, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count) return;
    if (falive[idx]) return;
    // Each dead food has 'rate' chance per 100 steps of respawning
    unsigned int rng = seed + idx * 31 + step * 7;
    if (cr(&rng) < (float)rate/1000.0f) {
        fx[idx] = cr(&rng) * w;
        fy[idx] = cr(&rng) * w;
        falive[idx] = 1;
    }
}

int main(){
    int sizes[NSIZES]={128,256,512,1024};
    int regen=100; // 10% per 100 steps = 100/10000 per step
    printf("=== SCRIPT FOOD REGEN: Law 230 ===\nFood=%d Steps=%d Regen=10%%/100steps Trials=%d\n\n",FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_fx,FOOD*sizeof(float));cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    printf("%-8s | %-8s | %-10s | %-8s\n","Agents","PerAgent","FleetTot","Surv");
    printf("--------|----------|------------|----------\n");
    for(int s=0;s<NSIZES;s++){
        int n=sizes[s];
        cudaMalloc(&d_s,n*sizeof(float));cudaMalloc(&d_a,n*sizeof(int));
        int blk=(n+BLK-1)/BLK;int fblk=(FOOD+BLK-1)/BLK;
        float ts=0,ta=0,tt=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            // Run in chunks of 100 to allow regen
            for(int chunk=0;chunk<STEPS;chunk+=100){
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,100,n,FOOD,W,0,(unsigned int)(42+tr*1111+s*111+chunk*11));
                regen_food<<<fblk,BLK>>>(d_fx,d_fy,d_fa,FOOD,W,regen,chunk,(unsigned int)(42+tr*1111+s*111+chunk*11));
            }
            cudaDeviceSynchronize();
            float hs[n];int ha[n];cudaMemcpy(hs,d_s,n*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,n*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;float total=0;
            for(int i=0;i<n;i++){avg+=hs[i];ac+=ha[i];total+=hs[i];}
            ts+=avg/n;ta+=ac;tt+=total;
        }
        ts/=TRIALS;float surv=ta/TRIALS/n*100;tt/=TRIALS;
        printf("%-8d | %6.3f   | %8.1f   | %5.1f%%\n",n,ts,tt,surv);
        cudaFree(d_s);cudaFree(d_a);
    }
    printf("\n>> Law 230: Does food regen create fleet effect for scripts?\n");
    cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);
    return 0;
}