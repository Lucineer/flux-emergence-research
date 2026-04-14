// experiment-predator-prey.cu — Two species competing for same food
// Species A: fast+small grab (explorer) vs Species B: slow+large grab (exploiter)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_AGENTS 512
#define FOOD 300
#define STEPS 5000
#define WORLD 1024
#define BLOCK 256

// Use structs for clean state reset
typedef struct { float x,y,food; int seed; } Agent;
typedef struct { float x,y; int seed; } Food;

__device__ Agent a[MAX_AGENTS];
__device__ Food f[FOOD];
__device__ int na, nb, nf;
__device__ float grab_a, grab_b, speed_a, speed_b;

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a, float b) { float d=fabsf(a-b); return fminf(d,WORLD-d); }
__device__ float wr(float v) { float r=fmodf(v,WORLD); return r<0?r+WORLD:r; }

__global__ void init(int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < na) {
        a[i].seed = seed+i*137;
        a[i].x = cr(&a[i].seed)*WORLD;
        a[i].y = cr(&a[i].seed)*WORLD;
        a[i].food = 0;
    }
    if (i < nf) {
        f[i].seed = seed+50000+i*997;
        f[i].x = cr(&f[i].seed)*WORLD;
        f[i].y = cr(&f[i].seed)*WORLD;
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= na) return;
    
    float gr = (i < nb) ? grab_b : grab_a;  // first na are A, then B agents
    float sp = (i < nb) ? speed_b : speed_a;
    
    // Find nearest food
    float best_d=1e9, bfx=0, bfy=0;
    for (int j=0;j<nf;j++) {
        float dx=wd(a[i].x,f[j].x), dy=wd(a[i].y,f[j].y);
        float d=dx*dx+dy*dy;
        if (d<best_d) { best_d=d; bfx=f[j].x; bfy=f[j].y; }
    }
    
    // Move
    float mx=0,my=0;
    if (best_d < 2500.0f) { // perception=50
        float dx=wd(bfx,a[i].x), dy=wd(bfy,a[i].y);
        float d=sqrtf(best_d);
        if(d>0){mx=dx/d*sp; my=dy/d*sp;}
    } else {
        float ang=cr(&a[i].seed)*6.2832f;
        mx=cosf(ang)*sp; my=sinf(ang)*sp;
    }
    a[i].x=wr(a[i].x+mx);
    a[i].y=wr(a[i].y+my);
    
    // Collect
    for (int j=0;j<nf;j++) {
        float dx=wd(a[i].x,f[j].x), dy=wd(a[i].y,f[j].y);
        if (dx*dx+dy*dy < gr*gr) {
            a[i].food++;
            f[j].x=cr(&f[j].seed)*WORLD;
            f[j].y=cr(&f[j].seed)*WORLD;
            break;
        }
    }
}

void run(const char* label, int _na, int _nb, float ga, float gb, float sa, float sb, int _nf, int seed) {
    cudaMemcpyToSymbol(na, &_na, sizeof(int));
    cudaMemcpyToSymbol(nb, &_nb, sizeof(int));
    cudaMemcpyToSymbol(nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(grab_a, &ga, sizeof(float));
    cudaMemcpyToSymbol(grab_b, &gb, sizeof(float));
    cudaMemcpyToSymbol(speed_a, &sa, sizeof(float));
    cudaMemcpyToSymbol(speed_b, &sb, sizeof(float));
    
    int total = _na + _nb;
    int blocks = (total + BLOCK - 1) / BLOCK;
    int fblocks = (_nf + BLOCK - 1) / BLOCK;
    
    init<<<max(blocks,fblocks), BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    Agent h[MAX_AGENTS];
    cudaMemcpyFromSymbol(h, a, sizeof(Agent)*MAX_AGENTS);
    
    float ta=0, tb=0;
    for (int i=0;i<_nb;i++) tb+=h[i].food; // B agents first
    for (int i=_nb;i<total;i++) ta+=h[i].food; // A agents after
    
    printf("  %-40s A/agent=%.1f  B/agent=%.1f  total=%.0f\n",
           label, _na>0?ta/_na:0, _nb>0?tb/_nb:0, ta+tb);
}

int main() {
    printf("=== Two Species Competition ===\n");
    printf("A: fast+small grab (explorer)  B: slow+large grab (exploiter)\n\n");
    
    printf("--- Baseline ---\n");
    run("All A: grab=12, speed=5", 512, 0, 12, 0, 5, 0, 300, 42);
    run("All B: grab=25, speed=2", 0, 512, 0, 25, 0, 2, 300, 42);
    run("Balanced: grab=15, speed=3", 512, 0, 15, 0, 3, 0, 300, 42);
    
    printf("\n--- Mixed (equal split) ---\n");
    run("50%% A / 50%% B", 256, 256, 12, 25, 5, 2, 300, 42);
    run("75%% A / 25%% B", 384, 128, 12, 25, 5, 2, 300, 42);
    run("25%% A / 75%% B", 128, 384, 12, 25, 5, 2, 300, 42);
    
    printf("\n--- Extreme specialization ---\n");
    run("50/50 extreme (8/speed=7 vs 30/speed=1)", 256, 256, 8, 30, 7, 1, 300, 42);
    run("50/50 mild (13/speed=4 vs 18/speed=2.5)", 256, 256, 13, 18, 4, 2.5, 300, 42);
    
    printf("\n--- Scarcity (100 food) ---\n");
    run("All A, 100 food", 512, 0, 12, 0, 5, 0, 100, 42);
    run("All B, 100 food", 0, 512, 0, 25, 0, 2, 100, 42);
    run("50/50, 100 food", 256, 256, 12, 25, 5, 2, 100, 42);
    
    printf("\n--- Grab range sweep (all same species) ---\n");
    run("All grab=8", 512, 0, 8, 0, 3, 0, 300, 42);
    run("All grab=15", 512, 0, 15, 0, 3, 0, 300, 42);
    run("All grab=25", 512, 0, 25, 0, 3, 0, 300, 42);
    run("All grab=40", 512, 0, 40, 0, 3, 0, 300, 42);
    
    return 0;
}
