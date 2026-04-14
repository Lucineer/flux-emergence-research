// experiment-dcs-swarm.cu — DCS protocol at mega-swarm scale
// Does DCS scale? Or does stampede get worse with more agents?
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define FOOD 400
#define STEPS 3000
#define WORLD 1024
#define BLOCK 256
#define GUILD_TYPES 3
#define MAX_AGENTS 8192

__device__ float ax[MAX_AGENTS], ay[MAX_AGENTS], a_food[MAX_AGENTS];
__device__ int a_seed[MAX_AGENTS], a_guild[MAX_AGENTS];
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD];
__device__ int d_n, d_nf, d_dcs, d_ttl;
__device__ float d_grab, d_speed, d_perc;
// DCS with TTL
__device__ float g_best_x[GUILD_TYPES], g_best_y[GUILD_TYPES];
__device__ int g_found[GUILD_TYPES], g_age[GUILD_TYPES]; // TTL counter

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
    if (i < GUILD_TYPES) { g_found[i]=0; g_age[i]=999; }
}

__global__ void step(int step_num) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    
    float mx=0,my=0;
    
    // Check DCS with TTL
    if (d_dcs && g_found[a_guild[i]] && g_age[a_guild[i]] < d_ttl) {
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
                g_age[a_guild[i]]=0; // refresh TTL
            }
            fx[j]=cr(&f_seed[j])*WORLD; fy[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

__global__ void age_dcs() {
    for (int i=0;i<GUILD_TYPES;i++) g_age[i]++;
}

void run(const char* label, int dcs, int ttl, int n_agents, int seed) {
    int _n=n_agents, _nf=FOOD;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(d_dcs, &dcs, sizeof(int));
    cudaMemcpyToSymbol(d_ttl, &ttl, sizeof(int));
    
    int blocks=(n_agents+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    for (int s=0;s<STEPS;s++) {
        step<<<blocks,BLOCK>>>(s);
        if (dcs) age_dcs<<<1,1>>>();
        cudaDeviceSynchronize();
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    float h_food[MAX_AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*n_agents);
    float total=0;
    for (int i=0;i<n_agents;i++) total+=h_food[i];
    printf("  %-55s agents=%5d  per=%8.1f  %.0fms\n",
           label, n_agents, total/n_agents, ms);
}

int main() {
    printf("=== DCS at Scale — TTL Invalidation ===\n\n");
    
    printf("--- No DCS baseline ---\n");
    run("No DCS", 0, 0, 512, 42);
    run("No DCS", 0, 0, 2048, 42);
    run("No DCS", 0, 0, 4096, 42);
    run("No DCS", 0, 0, 8192, 42);
    
    printf("\n--- DCS with TTL=5 (invalidate after 5 steps) ---\n");
    run("DCS TTL=5", 1, 5, 512, 42);
    run("DCS TTL=5", 1, 5, 2048, 42);
    run("DCS TTL=5", 1, 5, 4096, 42);
    run("DCS TTL=5", 1, 5, 8192, 42);
    
    printf("\n--- DCS with TTL=20 ---\n");
    run("DCS TTL=20", 1, 20, 512, 42);
    run("DCS TTL=20", 1, 20, 2048, 42);
    run("DCS TTL=20", 1, 20, 4096, 42);
    run("DCS TTL=20", 1, 20, 8192, 42);
    
    printf("\n--- DCS with TTL=100 (nearly permanent) ---\n");
    run("DCS TTL=100", 1, 100, 512, 42);
    run("DCS TTL=100", 1, 100, 2048, 42);
    run("DCS TTL=100", 1, 100, 4096, 42);
    run("DCS TTL=100", 1, 100, 8192, 42);
    
    printf("\n--- DCS no TTL (stampede mode) ---\n");
    run("DCS no TTL", 1, 99999, 512, 42);
    run("DCS no TTL", 1, 99999, 2048, 42);
    run("DCS no TTL", 1, 99999, 4096, 42);
    
    return 0;
}
