#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSIZES 6
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

int main(){
    int sizes[NSIZES]={32,64,128,256,512,1024};
    printf("=== FLEET SIZE OPTIMAL: Law 250 ===\nFood=%d Steps=%d Trials=%d\n\n",FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_fx,FOOD*sizeof(float));cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    printf("%-8s | %-8s | %-8s | %-10s\n","Agents","PerAgent","Fleet","Efficiency");
    printf("--------|----------|----------|------------\n");
    float best_eff=0;int bi=0;
    for(int s=0;s<NSIZES;s++){
        int n=sizes[s];
        float *d_sc;int *d_al;cudaMalloc(&d_sc,n*sizeof(float));cudaMalloc(&d_al,n*sizeof(int));
        int blk=(n+BLK-1)/BLK;
        float ts=0,tt=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_sc,d_al,d_fx,d_fy,d_fa,STEPS,n,FOOD,W,(unsigned int)(42+tr*1111+s*111));
            cudaDeviceSynchronize();
            float hs[n];cudaMemcpy(hs,d_sc,n*sizeof(float),cudaMemcpyDeviceToHost);
            float avg=0;float total=0;for(int i=0;i<n;i++){avg+=hs[i];total+=hs[i];}
            ts+=avg/n;tt+=total;
        }
        ts/=TRIALS;tt/=TRIALS;
        float eff=tt/n; // fleet total per agent = total food per agent deployed
        if(eff>best_eff){best_eff=eff;bi=s;}
        printf("%-8d | %6.3f   | %6.1f   | %6.3f\n",n,ts,tt,eff);
        cudaFree(d_sc);cudaFree(d_al);
    }
    printf("\nBest efficiency: %d agents (%.3f food/agent deployed)\n",sizes[bi],best_eff);
    cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);
    return 0;
}