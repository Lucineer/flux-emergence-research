// experiment-swarm-size.cu — Agent count sweep: 128→4096
// Casey asked: shorter sessions + larger swarms?
// Test: how does agent count affect per-agent fitness?
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define FOOD 200
#define STEPS 2000
#define WORLD 1024
#define BLOCK 256
#define MAX_AGENTS 4096

__device__ float ax[MAX_AGENTS], ay[MAX_AGENTS], a_food[MAX_AGENTS];
__device__ int a_seed[MAX_AGENTS];
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD];
__device__ int d_n, d_nf;
__device__ float d_grab, d_speed, d_perc;

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a,float b) { float d=fabsf(a-b); return fminf(d,WORLD-d); }
__device__ float wr(float v) { float r=fmodf(v,WORLD); return r<0?r+WORLD:r; }

__global__ void init(int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    a_seed[i] = seed+i*137;
    ax[i] = cr(&a_seed[i])*WORLD;
    ay[i] = cr(&a_seed[i])*WORLD;
    a_food[i] = 0;
    if (i < d_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD;
        fy[i] = cr(&f_seed[i])*WORLD;
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    float best_d=1e9,bfx=0,bfy=0;
    for (int j=0;j<d_nf;j++) {
        float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
        float d=dx*dx+dy*dy;
        if(d<best_d){best_d=d;bfx=fx[j];bfy=fy[j];}
    }
    float mx=0,my=0;
    if (best_d < d_perc*d_perc) {
        float dx=wd(bfx,ax[i]),dy=wd(bfy,ay[i]);
        float d=sqrtf(best_d); if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
    } else {
        float ang=cr(&a_seed[i])*6.2832f;
        mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
    }
    ax[i]=wr(ax[i]+mx); ay[i]=wr(ay[i]+my);
    for (int j=0;j<d_nf;j++) {
        float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
        if (dx*dx+dy*dy < d_grab*d_grab) {
            a_food[i]++;
            fx[j]=cr(&f_seed[j])*WORLD; fy[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

void run(int n_agents, int seed) {
    int _n=n_agents, _nf=FOOD;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    
    int blocks=(n_agents+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[MAX_AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*n_agents);
    float total=0;
    for (int i=0;i<n_agents;i++) total+=h_food[i];
    printf("  agents=%5d  total=%10.0f  per=%8.1f  density=%.4f\n",
           n_agents, total, total/n_agents, (float)n_agents/(1024.0f*1024.0f));
}

int main() {
    printf("=== Swarm Size Sweep ===\n");
    printf("200 food, 2000 steps, 1024x1024 world\n\n");
    
    printf("--- Agent count sweep (fixed food=200) ---\n");
    run(32, 42);
    run(64, 42);
    run(128, 42);
    run(256, 42);
    run(512, 42);
    run(1024, 42);
    run(2048, 42);
    run(4096, 42);
    
    return 0;
}
