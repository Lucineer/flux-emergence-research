#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NFRACTIONS 7
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w,
    int scripted_count, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    int is_scripted = (tid < scripted_count) ? 1 : 0;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    for (int i=0;i<8;i++) script_dir[i] = base_angle + i*0.785f;
    for (int t=0;t<steps&&energy>0;t++){
        float dx=0,dy=0,r=4.0f;
        if(is_scripted){
            int p=t%8;dx=cosf(script_dir[p])*2.0f;dy=sinf(script_dir[p])*2.0f;
        } else {
            dx=(cr(&rng)-0.5f)*6.0f*speed_mult;
            dy=(cr(&rng)-0.5f)*6.0f*speed_mult;
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
    float fractions[NFRACTIONS]={0.0f,0.1f,0.25f,0.5f,0.75f,0.9f,1.0f};
    float test_speeds[3]={4.0f,6.0f,8.0f};
    printf("=== MIXED FLEET AT MODERATE SPEED: Law 206 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    for(int sp=0;sp<3;sp++){
        printf("--- Speed %dx ---\n",(int)test_speeds[sp]);
        printf("%-10s | %-8s | %-8s\n","Fraction","Score","Survive");
        printf("-----------|----------|----------\n");
        float best=0;int bf=0;
        for(int f=0;f<NFRACTIONS;f++){
            int sc=(int)(N*fractions[f]);
            float ts=0,ta=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,test_speeds[sp],STEPS,N,FOOD,W,sc,(unsigned int)(42+tr*1111+f*111+sp*11));
                cudaDeviceSynchronize();
                float hs[N];int ha[N];
                cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
                float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
                ts+=avg/N;ta+=ac;
            }
            ts/=TRIALS;ta=ta/TRIALS/N*100;
            if(ts>best){best=ts;bf=f;}
            printf("%-10.0f%% | %6.3f   | %5.1f%%\n",fractions[f]*100,ts,ta);
        }
        printf("Best: %.0f%% scripted (score=%.3f)\n\n",fractions[bf]*100,best);
    }
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}