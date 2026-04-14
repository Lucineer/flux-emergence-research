// experiment-herding.cu — Do agents following crowd movement outperform independent agents?
// Hypothesis: when most agents move in one direction, followers benefit (information cascade)
// Counter: herding creates congestion, reduces total (Law 3 info under scarcity)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 512
#define FOOD 200
#define STEPS 5000
#define WORLD 1024
#define BLOCK 256

__device__ int cfg_n, cfg_nf, cfg_steps;
__device__ float cfg_grab, cfg_speed, cfg_perc, cfg_herd;
// 0 = no herding, 0.1 = slight bias toward crowd direction, 1.0 = full follow

__device__ float ax[AGENTS], ay[AGENTS], a_food[AGENTS], a_vx[AGENTS], a_vy[AGENTS];
__device__ int a_seed[AGENTS];
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD];

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a,float b) { float d=fabsf(a-b); return fminf(d,WORLD-d); }
__device__ float wr(float v) { float r=fmodf(v,WORLD); return r<0?r+WORLD:r; }

__global__ void init(int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= cfg_n) return;
    a_seed[i] = seed+i*137;
    ax[i] = cr(&a_seed[i])*WORLD;
    ay[i] = cr(&a_seed[i])*WORLD;
    a_food[i] = 0;
    a_vx[i] = 0; a_vy[i] = 0;
    if (i < cfg_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD;
        fy[i] = cr(&f_seed[i])*WORLD;
    }
}

// Compute average movement direction of nearby agents
__global__ void compute_herd() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= cfg_n) return;
    
    float hx=0, hy=0, count=0;
    float perc2 = 100.0f*100.0f; // perception radius for herding = 100
    
    for (int j=0; j<cfg_n; j++) {
        if (i==j) continue;
        float dx=wd(ax[i],ax[j]), dy=wd(ay[i],ay[j]);
        if (dx*dx+dy*dy < perc2) {
            hx += a_vx[j];
            hy += a_vy[j];
            count++;
        }
    }
    
    if (count > 0) {
        a_vx[i] = (1-cfg_herd)*a_vx[i] + cfg_herd*(hx/count);
        a_vy[i] = (1-cfg_herd)*a_vy[i] + cfg_herd*(hy/count);
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= cfg_n) return;
    
    // Find nearest food
    float best_d=1e9, bfx=0, bfy=0;
    for (int j=0;j<cfg_nf;j++) {
        float dx=wd(ax[i],fx[j]), dy=wd(ay[i],fy[j]);
        float d=dx*dx+dy*dy;
        if (d<best_d) { best_d=d; bfx=fx[j]; bfy=fy[j]; }
    }
    
    float mx=0, my=0;
    if (best_d < cfg_perc*cfg_perc) {
        float dx=wd(bfx,ax[i]), dy=wd(bfy,ay[i]);
        float d=sqrtf(best_d);
        if(d>0){mx=dx/d*cfg_speed; my=dy/d*cfg_speed;}
    } else {
        float ang=cr(&a_seed[i])*6.2832f;
        mx=cosf(ang)*cfg_speed; my=sinf(ang)*cfg_speed;
    }
    
    // Blend with herd direction
    mx = (1-cfg_herd)*mx + cfg_herd*a_vx[i];
    my = (1-cfg_herd)*my + cfg_herd*a_vy[i];
    
    // Store velocity for herd computation
    a_vx[i] = mx;
    a_vy[i] = my;
    
    ax[i] = wr(ax[i]+mx);
    ay[i] = wr(ay[i]+my);
    
    // Collect
    for (int j=0;j<cfg_nf;j++) {
        float dx=wd(ax[i],fx[j]), dy=wd(ay[i],fy[j]);
        if (dx*dx+dy*dy < cfg_grab*cfg_grab) {
            a_food[i]++;
            fx[j]=cr(&f_seed[j])*WORLD;
            fy[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

void run(const char* label, float herd, int nf, int seed) {
    int _n=AGENTS; float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(cfg_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(cfg_nf, &nf, sizeof(int));
    cudaMemcpyToSymbol(cfg_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(cfg_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(cfg_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(cfg_herd, &herd, sizeof(float));
    
    int blocks = (AGENTS+BLOCK-1)/BLOCK;
    int fblocks = (nf+BLOCK-1)/BLOCK;
    
    init<<<max(blocks,fblocks), BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) {
        if (herd > 0) compute_herd<<<blocks,BLOCK>>>();
        step<<<blocks,BLOCK>>>();
        cudaDeviceSynchronize();
    }
    
    float h_food[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*AGENTS);
    float total=0;
    for (int i=0;i<AGENTS;i++) total+=h_food[i];
    printf("  %-40s total=%.0f  per_agent=%.1f\n", label, total, total/AGENTS);
}

int main() {
    printf("=== Herding / Information Cascade ===\n");
    printf("Does following the crowd help or hurt?\n\n");
    
    printf("--- Normal food (200) ---\n");
    run("No herding (control)", 0.0f, 200, 42);
    run("Herd 0.1 (slight)", 0.1f, 200, 42);
    run("Herd 0.3", 0.3f, 200, 42);
    run("Herd 0.5", 0.5f, 200, 42);
    run("Herd 0.7", 0.7f, 200, 42);
    run("Herd 0.9", 0.9f, 200, 42);
    
    printf("\n--- Scarcity (50 food) ---\n");
    run("No herding, 50 food", 0.0f, 50, 42);
    run("Herd 0.3, 50 food", 0.3f, 50, 42);
    run("Herd 0.5, 50 food", 0.5f, 50, 42);
    run("Herd 0.9, 50 food", 0.9f, 50, 42);
    
    printf("\n--- Abundance (800 food) ---\n");
    run("No herding, 800 food", 0.0f, 800, 42);
    run("Herd 0.3, 800 food", 0.3f, 800, 42);
    run("Herd 0.9, 800 food", 0.9f, 800, 42);
    
    printf("\n=== Analysis ===\n");
    printf("If herding hurts at all levels: confirms Law 3 (info only matters under scarcity)\n");
    printf("If herding helps in abundance: information cascades are valuable when resources plentiful\n");
    printf("If herding helps in scarcity: contradicts Law 3, herding IS a scarcity adaptation\n");
    
    return 0;
}
