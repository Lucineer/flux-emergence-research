// experiment-perception-dcs.cu — Perception vs DCS tradeoff
// At what point does DCS make individual perception irrelevant?
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 512
#define FOOD 200
#define STEPS 5000
#define WORLD 1024
#define BLOCK 256
#define GUILD_TYPES 3

__device__ float ax[AGENTS], ay[AGENTS], a_food[AGENTS];
__device__ int a_seed[AGENTS], a_guild[AGENTS];
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD];
__device__ int d_n, d_nf, d_dcs;
__device__ float d_grab, d_speed, d_perc;
__device__ float g_best_x[GUILD_TYPES], g_best_y[GUILD_TYPES];
__device__ int g_found[GUILD_TYPES];

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
    a_guild[i] = i % GUILD_TYPES;
    if (i < d_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD;
        fy[i] = cr(&f_seed[i])*WORLD;
    }
    if (i < GUILD_TYPES) { g_found[i] = 0; }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    
    float mx=0,my=0;
    
    if (d_dcs && g_found[a_guild[i]]) {
        float dx=wd(g_best_x[a_guild[i]],ax[i]);
        float dy=wd(g_best_y[a_guild[i]],ay[i]);
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
    } else {
        float best_d=1e9,bfx=0,bfy=0;
        for (int j=0;j<d_nf;j++) {
            float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
            float d=dx*dx+dy*dy;
            if(d<best_d){best_d=d;bfx=fx[j];bfy=fy[j];}
        }
        if (best_d < d_perc*d_perc) {
            float dx=wd(bfx,ax[i]),dy=wd(bfy,ay[i]);
            float d=sqrtf(best_d);
            if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
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
            if (d_dcs) {
                g_best_x[a_guild[i]]=fx[j];
                g_best_y[a_guild[i]]=fy[j];
                g_found[a_guild[i]]=1;
            }
            fx[j]=cr(&f_seed[j])*WORLD;
            fy[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

void run(const char* label, int dcs, float perc, int seed) {
    int _n=AGENTS, _nf=FOOD;
    float _g=15.0f, _sp=3.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &perc, sizeof(float));
    cudaMemcpyToSymbol(d_dcs, &dcs, sizeof(int));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*AGENTS);
    float total=0;
    for (int i=0;i<AGENTS;i++) total+=h_food[i];
    printf("  %-40s total=%.0f  per=%.1f\n", label, total, total/AGENTS);
}

int main() {
    printf("=== Perception vs DCS Tradeoff ===\n\n");
    
    printf("--- No DCS, perception sweep ---\n");
    run("No DCS, perc=10", 0, 10.0f, 42);
    run("No DCS, perc=25", 0, 25.0f, 42);
    run("No DCS, perc=50", 0, 50.0f, 42);
    run("No DCS, perc=100", 0, 100.0f, 42);
    run("No DCS, perc=200", 0, 200.0f, 42);
    run("No DCS, perc=500", 0, 500.0f, 42);
    
    printf("\n--- With DCS, perception sweep ---\n");
    run("DCS, perc=10", 1, 10.0f, 42);
    run("DCS, perc=25", 1, 25.0f, 42);
    run("DCS, perc=50", 1, 50.0f, 42);
    run("DCS, perc=100", 1, 100.0f, 42);
    run("DCS, perc=200", 1, 200.0f, 42);
    run("DCS, perc=500", 1, 500.0f, 42);
    
    printf("\n--- Food count sweep (perc=50) ---\n");
    int _nf_save;
    cudaMemcpyFromSymbol(&_nf_save, d_nf, sizeof(int));
    for (int nf = 25; nf <= 800; nf *= 2) {
        char l1[64], l2[64];
        snprintf(l1, 64, "No DCS, %d food", nf);
        snprintf(l2, 64, "DCS, %d food", nf);
        run(l1, 0, 50.0f, 42);
        run(l2, 1, 50.0f, 42);
    }
    
    return 0;
}
