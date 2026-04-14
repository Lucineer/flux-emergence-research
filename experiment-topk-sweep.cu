// experiment-topk-sweep.cu — Find optimal TOP-K for multi-point DCS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 4096
#define FOOD 400
#define STEPS 3000
#define WORLD 1024
#define BLOCK 256
#define GUILD_TYPES 3
#define MAX_K 32

__device__ float ax[AGENTS], ay[AGENTS], a_food[AGENTS];
__device__ int a_seed[AGENTS], a_guild[AGENTS];
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD];
__device__ int d_n, d_nf, d_topk;
__device__ float d_grab, d_speed, d_perc;
__device__ float g_mx[GUILD_TYPES*MAX_K], g_my[GUILD_TYPES*MAX_K];
__device__ int g_mcount[GUILD_TYPES];

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a,float b) { float d=fabsf(a-b); return fminf(d,WORLD-d); }
__device__ float wr(float v) { float r=fmodf(v,WORLD); return r<0?r+WORLD:r; }

__global__ void init(int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    a_seed[i] = seed+i*137;
    ax[i] = cr(&a_seed[i])*WORLD; ay[i] = cr(&a_seed[i])*WORLD;
    a_food[i] = 0; a_guild[i] = i % GUILD_TYPES;
    if (i < d_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD; fy[i] = cr(&f_seed[i])*WORLD;
    }
    if (i < GUILD_TYPES) {
        g_mcount[i]=0;
        for (int k=0;k<MAX_K;k++) { g_mx[i*MAX_K+k]=0; g_my[i*MAX_K+k]=0; }
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    float mx=0,my=0;
    int g=a_guild[i], cnt=g_mcount[g];
    
    if (cnt > 0) {
        float best_d=1e9,bgx=0,bgy=0;
        for (int k=0;k<cnt && k<d_topk;k++) {
            float dx=wd(g_mx[g*MAX_K+k],ax[i]),dy=wd(g_my[g*MAX_K+k],ay[i]);
            float d=dx*dx+dy*dy;
            if(d<best_d){best_d=d;bgx=g_mx[g*MAX_K+k];bgy=g_my[g*MAX_K+k];}
        }
        if (best_d < d_perc*d_perc) {
            float dx=wd(bgx,ax[i]),dy=wd(bgy,ay[i]);
            float d=sqrtf(best_d); if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
        } else {
            float ang=cr(&a_seed[i])*6.2832f;
            mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
        }
    } else {
        float best_d=1e9,bfx=0,bfy=0;
        for (int j=0;j<d_nf;j++) {
            float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
            float d=dx*dx+dy*dy;
            if(d<best_d){best_d=d;bfx=fx[j];bfy=fy[j];}
        }
        if (best_d < d_perc*d_perc) {
            float dx=wd(bfx,ax[i]),dy=wd(bfy,ay[i]);
            float d=sqrtf(best_d); if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
        } else {
            float ang=cr(&a_seed[i])*6.2832f;
            mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
        }
    }
    
    ax[i]=wr(ax[i]+mx); ay[i]=wr(ay[i]+my);
    for (int j=0;j<d_nf;j++) {
        float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
        if (dx*dx+dy*dy < d_grab*d_grab) {
            a_food[i]++;
            int slot = g_mcount[g] % MAX_K;
            g_mx[g*MAX_K+slot]=fx[j]; g_my[g*MAX_K+slot]=fy[j];
            if (g_mcount[g] < MAX_K) g_mcount[g]++;
            fx[j]=cr(&f_seed[j])*WORLD; fy[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

void run(int topk, int seed) {
    int _n=AGENTS, _nf=FOOD;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(d_topk, &topk, sizeof(int));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*AGENTS);
    float total=0;
    for (int i=0;i<AGENTS;i++) total+=h_food[i];
    printf("  TOP-K=%2d  total=%10.0f  per=%8.1f\n", topk, total, total/AGENTS);
}

int main() {
    printf("=== TOP-K Sweep — Optimal Multi-Point DCS ===\n");
    printf("4096 agents, 400 food, 3000 steps\n\n");
    
    printf("--- TOP-K sweep ---\n");
    run(1, 42);   // single point (stampede)
    run(2, 42);
    run(4, 42);
    run(8, 42);
    run(12, 42);
    run(16, 42);
    run(20, 42);
    run(24, 42);
    run(32, 42);
    
    return 0;
}
