#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NF 128
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores_a, float *scores_b, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w,
    float noise_a, float noise_b, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    // Fleet A
    {
        unsigned int rng = seed + tid * 997;
        float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
        float base_angle = tid * 2.39996f;
        float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
        for(int t=0;t<steps&&energy>0;t++){
            int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
            dx+=(cr(&rng)-0.5f)*2.0f*noise_a;
            dy+=(cr(&rng)-0.5f)*2.0f*noise_a;
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
        scores_a[tid]=score;
    }
    // Fleet B
    {
        unsigned int rng = seed + (tid+n) * 997;
        float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
        float base_angle = (tid+n) * 2.39996f;
        float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
        for(int t=0;t<steps&&energy>0;t++){
            int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
            dx+=(cr(&rng)-0.5f)*2.0f*noise_b;
            dy+=(cr(&rng)-0.5f)*2.0f*noise_b;
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
        scores_b[tid]=score;
    }
}

int main(){
    printf("=== NOISE IN COMPETITION: Law 246 ===\nFleet=%d Food=%d Steps=%d Trials=%d\n\n",NF,FOOD,STEPS,TRIALS);
    float *d_sa,*d_sb,*d_fx,*d_fy;int *d_fa;
    cudaMalloc(&d_sa,NF*sizeof(float));cudaMalloc(&d_sb,NF*sizeof(float));
    cudaMalloc(&d_fx,FOOD*sizeof(float));cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(NF+BLK-1)/BLK;
    float na_arr[]={0.3f,0.3f,1.0f};
    float nb_arr[]={0.0f,1.0f,0.0f};
    const char* nm[]={"Noise0.3 vs Pure","Noise0.3 vs Noise1.0","Noise1.0 vs Pure"};
    for(int m=0;m<3;m++){
        float ta=0,tb=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_sa,d_sb,d_fx,d_fy,d_fa,STEPS,NF,FOOD,W,na_arr[m],nb_arr[m],(unsigned int)(42+tr*1111+m*111));
            cudaDeviceSynchronize();
            float hsa[NF],hsb[NF];cudaMemcpy(hsa,d_sa,NF*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hsb,d_sb,NF*sizeof(float),cudaMemcpyDeviceToHost);
            float avga=0,avgb=0;for(int i=0;i<NF;i++){avga+=hsa[i];avgb+=hsb[i];}
            ta+=avga/NF;tb+=avgb/NF;
        }
        ta/=TRIALS;tb/=TRIALS;
        printf("%-28s: A=%.3f B=%.3f ratio=%.2f\n",nm[m],ta,tb,ta/(tb+0.001f));
    }
    printf("\n>> Law 246: Does noise give competitive advantage?\n");
    cudaFree(d_sa);cudaFree(d_sb);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);
    return 0;
}