#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSPEEDS 9
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w,
    int strat, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    for(int i=0;i<8;i++) script_dir[i]=base_angle+i*0.785f;
    
    for(int t=0;t<steps&&energy>0;t++){
        float dx=0,dy=0,r=4.0f;
        if(strat==0){ // Random walk
            dx=(cr(&rng)-0.5f)*6.0f*speed_mult;
            dy=(cr(&rng)-0.5f)*6.0f*speed_mult;
        } else if(strat==1){ // Food scan — approach nearest food
            float best_d=1e10f,bx=x,by=y;
            // SCAN ALL FOOD — not every 8th
            for(int i=0;i<food_count;i++){
                if(!falive[i])continue;
                float fdx=fx[i]-x,fdy=fy[i]-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=fdx*fdx+fdy*fdy;
                if(d<best_d){best_d=d;bx=fx[i];by=fy[i];}
            }
            if(best_d<1600.0f){ // detection range 40 units
                // Approach at FIXED speed (not speed-scaled) to avoid overshoot
                float fdx=bx-x,fdy=by-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=sqrtf(fdx*fdx+fdy*fdy);
                if(d>0.1f){dx=fdx/d*2.0f;dy=fdy/d*2.0f;}
            } else {
                dx=(cr(&rng)-0.5f)*4.0f;
                dy=(cr(&rng)-0.5f)*4.0f;
            }
        } else { // Scripted
            int p=t%8;dx=cosf(script_dir[p])*2.0f;dy=sinf(script_dir[p])*2.0f;
        }
        float dist=sqrtf(dx*dx+dy*dy);
        energy-=0.005f+dist*0.003f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        for(int i=0;i<food_count;i++){
            if(!falive[i])continue;
            float fdx=fx[i]-x,fdy=fy[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            if(fdx*fdx+fdy*fdy<r*r){int old=atomicExch(&falive[i],0);if(old){energy=fminf(energy+10.0f,200.0f);score+=1.0f;}}
        }
    }
    scores[tid]=score;alive[tid]=(energy>0)?1:0;
}

int main(){
    const char* nm[]={"Random","FoodScan","Scripted"};
    float speeds[NSPEEDS]={1,2,4,6,8,10,12,16,24};
    printf("=== FOOD SCAN CROSSOVER (FIXED): Law 209 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n",N,FOOD,STEPS,TRIALS);
    printf("Fix: scan ALL food, fixed approach speed 2.0\n\n");
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    for(int s=0;s<NSPEEDS;s++){
        float res[3]={0};
        int surv[3]={0};
        for(int st=0;st<3;st++){
            float ts=0,ta=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,speeds[s],STEPS,N,FOOD,W,st,(unsigned int)(42+tr*1111+s*111+st*11));
                cudaDeviceSynchronize();
                float hs[N];int ha[N];
                cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
                float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
                ts+=avg/N;ta+=ac;
            }
            res[st]=ts/TRIALS;surv[st]=ta/TRIALS/N*100;
        }
        int best=0;for(int st=1;st<3;st++)if(res[st]>res[best])best=st;
        printf("%-4dx | Rand:%5.3f(%2d%%) | Scan:%5.3f(%2d%%) | Script:%5.3f(%2d%%) | Best:%s\n",
            (int)speeds[s],res[0],surv[0],res[1],surv[1],res[2],surv[2],nm[best]);
    }
    printf("\n>> Law 209: Food scanning with fixed approach speed\n");
    printf("   Key question: does perception overhead outweigh food-finding benefit?\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}