#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NBUDGETS 8
#define NMODES 2
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float start_energy, int steps, int n, int food_count, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = start_energy, score = 0.0f;
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
    unsigned int seed, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count || falive[idx]) return;
    unsigned int rng = seed + idx * 31 + step * 7;
    if (cr(&rng) < 0.001f) {
        fx[idx] = cr(&rng) * w;
        fy[idx] = cr(&rng) * w;
        falive[idx] = 1;
    }
}

int main(){
    float budgets[NBUDGETS]={10,20,30,40,50,60,70,80};
    const char* mn[]={"NoRegen","Regen"};
    printf("=== ENERGY x REGEN: Law 242 ===\nN=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    int blk=(N+BLK-1)/BLK;int fblk=(FOOD+BLK-1)/BLK;
    printf("%-8s | %-10s | %-10s | %-10s\n","Energy","NoRegen","Regen","Delta");
    printf("--------|------------|------------|------------\n");
    for(int b=0;b<NBUDGETS;b++){
        float tn=0,tr=0,sn=0,sr=0;
        for(int t=0;t<TRIALS;t++){
            // No regen
            cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,budgets[b],STEPS,N,FOOD,W,(unsigned int)(42+t*1111+b*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
            tn+=avg/N;sn+=ac;
            // Regen
            cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            for(int chunk=0;chunk<STEPS;chunk+=100){
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,budgets[b],100,N,FOOD,W,(unsigned int)(42+t*1111+b*111+chunk*11));
                regen_food<<<fblk,BLK>>>(d_fx,d_fy,d_fa,FOOD,W,(unsigned int)(42+t*1111+b*111),chunk);
            }
            cudaDeviceSynchronize();
            cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            avg=0;ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
            tr+=avg/N;sr+=ac;
        }
        tn/=TRIALS;tr/=TRIALS;float sn2=sn/TRIALS/N*100;float sr2=sr/TRIALS/N*100;
        printf("%-8.0f | %6.3f(%4.1f%%) | %6.3f(%4.1f%%) | %+.3f\n",budgets[b],tn,sn2,tr,sr2,tr-tn);
    }
    printf("\n>> Law 242: Does food regen lower energy cliff?\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}