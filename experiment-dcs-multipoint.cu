// experiment-dcs-multipoint.cu — DCS with distributed multi-point knowledge
// Law 24 says single-point DCS creates stampede.
// Fix: each guild stores TOP-K food locations, agents pick nearest.
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
#define TOP_K 8  // store top 8 food locations per guild

__device__ float ax[AGENTS], ay[AGENTS], a_food[AGENTS];
__device__ int a_seed[AGENTS], a_guild[AGENTS];
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD];
__device__ int d_n, d_nf, d_mode; // 0=none, 1=single-point, 2=multi-point TOP_K
__device__ float d_grab, d_speed, d_perc;

// Single-point DCS
__device__ float g_sx[GUILD_TYPES], g_sy[GUILD_TYPES];
__device__ int g_sf[GUILD_TYPES];

// Multi-point DCS
__device__ float g_mx[GUILD_TYPES*TOP_K], g_my[GUILD_TYPES*TOP_K];
__device__ int g_mcount[GUILD_TYPES];

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
    if (i < GUILD_TYPES) {
        g_sf[i]=0;
        g_mcount[i]=0;
        for (int k=0;k<TOP_K;k++) { g_mx[i*TOP_K+k]=0; g_my[i*TOP_K+k]=0; }
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    
    float mx=0,my=0;
    
    if (d_mode == 1 && g_sf[a_guild[i]]) {
        // Single-point DCS
        float dx=wd(g_sx[a_guild[i]],ax[i]),dy=wd(g_sy[a_guild[i]],ay[i]);
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
    } else if (d_mode == 2 && g_mcount[a_guild[i]] > 0) {
        // Multi-point DCS: pick nearest from guild's top-K
        float best_d=1e9,bgx=0,bgy=0;
        int g=a_guild[i], cnt=g_mcount[g];
        for (int k=0;k<cnt && k<TOP_K;k++) {
            float dx=wd(g_mx[g*TOP_K+k],ax[i]),dy=wd(g_my[g*TOP_K+k],ay[i]);
            float d=dx*dx+dy*dy;
            if(d<best_d){best_d=d;bgx=g_mx[g*TOP_K+k];bgy=g_my[g*TOP_K+k];}
        }
        if (best_d < d_perc*d_perc) {
            float dx=wd(bgx,ax[i]),dy=wd(bgy,ay[i]);
            float d=sqrtf(best_d); if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
        } else {
            float ang=cr(&a_seed[i])*6.2832f;
            mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
        }
    } else {
        // No DCS: direct perception
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
            int g=a_guild[i];
            if (d_mode == 1) {
                g_sx[g]=fx[j]; g_sy[g]=fy[j]; g_sf[g]=1;
            } else if (d_mode == 2) {
                int slot = g_mcount[g] % TOP_K;
                g_mx[g*TOP_K+slot]=fx[j]; g_my[g*TOP_K+slot]=fy[j];
                if (g_mcount[g] < TOP_K) g_mcount[g]++;
            }
            fx[j]=cr(&f_seed[j])*WORLD; fy[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

void run(const char* label, int mode, int seed) {
    int _n=AGENTS, _nf=FOOD;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(d_mode, &mode, sizeof(int));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*AGENTS);
    float total=0;
    for (int i=0;i<AGENTS;i++) total+=h_food[i];
    printf("  %-55s total=%10.0f  per=%8.1f\n", label, total, total/AGENTS);
}

int main() {
    printf("=== DCS Multi-Point vs Single-Point ===\n");
    printf("4096 agents, 400 food, 3000 steps\n\n");
    
    printf("--- Mode comparison ---\n");
    run("No DCS (direct perception)", 0, 42);
    run("Single-point DCS (stampede)", 1, 42);
    run("Multi-point DCS (TOP-8)", 2, 42);
    
    printf("\n--- Varying TOP-K ---\n");
    // Can't easily vary TOP-K at runtime (it's a compile-time constant)
    // But we can test different modes
    run("No DCS v2", 0, 99);
    run("Single-point v2", 1, 99);
    run("Multi-point TOP-8 v2", 2, 99);
    run("Multi-point TOP-8 v3", 2, 137);
    
    printf("\n=== Key Question ===\n");
    printf("Does multi-point DCS fix the stampede?\n");
    printf("If multi-point >> no DCS: distributed knowledge > direct perception\n");
    printf("If multi-point ≈ no DCS: DCS adds nothing over perception\n");
    printf("If multi-point < no DCS: DCS still hurts (knowledge lag)\n");
    
    return 0;
}
