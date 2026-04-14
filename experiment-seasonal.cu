// experiment-seasonal.cu — Pulsed food availability (feast/famine cycles)
// Tests whether agents conserve energy during famine, exploit during feast
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 512
#define FOOD_MAX 300
#define STEPS 5000
#define WORLD 1024
#define BLOCK 256

__device__ float d_ax[AGENTS], d_ay[AGENTS], d_food[AGENTS], d_energy[AGENTS];
__device__ int d_seed[AGENTS];
__device__ float d_fx[FOOD_MAX], d_fy[FOOD_MAX];
__device__ int d_fseed[FOOD_MAX], d_nf_actual;
__device__ int d_n, d_nf;
__device__ float d_grab, d_speed, d_perc;
__device__ float d_move_cost, d_energy_gain, d_max_energy;
__device__ int d_season_period, d_feast_fraction; // e.g. period=500, feast=250

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a,float b) { float d=fabsf(a-b); return fminf(d,WORLD-d); }
__device__ float wr(float v) { float r=fmodf(v,WORLD); return r<0?r+WORLD:r; }

__global__ void init(int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    d_seed[i] = seed+i*137;
    d_ax[i] = cr(&d_seed[i])*WORLD;
    d_ay[i] = cr(&d_seed[i])*WORLD;
    d_food[i] = 0;
    d_energy[i] = d_max_energy;
    if (i < FOOD_MAX) {
        d_fseed[i] = seed+50000+i*997;
        d_fx[i] = cr(&d_fseed[i])*WORLD;
        d_fy[i] = cr(&d_fseed[i])*WORLD;
    }
}

__global__ void step(int step_num) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    if (d_energy[i] <= 0) return;
    
    // Determine if feast or famine
    int phase = step_num % d_season_period;
    int is_feast = (phase < d_feast_fraction) ? 1 : 0;
    int active_food = is_feast ? d_nf : 0;
    
    float best_d=1e9, bfx=0, bfy=0;
    for (int j=0;j<active_food;j++) {
        float dx=wd(d_ax[i],d_fx[j]),dy=wd(d_ay[i],d_fy[j]);
        float d=dx*dx+dy*dy;
        if(d<best_d){best_d=d;bfx=d_fx[j];bfy=d_fy[j];}
    }
    
    float mx=0,my=0;
    if (best_d < d_perc*d_perc && active_food > 0) {
        float dx=wd(bfx,d_ax[i]),dy=wd(bfy,d_ay[i]);
        float d=sqrtf(best_d); if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
    } else {
        // During famine: don't move (conserve energy) vs random walk
        float ang=cr(&d_seed[i])*6.2832f;
        mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
    }
    
    d_energy[i] -= sqrtf(mx*mx+my*my) * d_move_cost;
    if (d_energy[i] <= 0) return;
    d_ax[i]=wr(d_ax[i]+mx); d_ay[i]=wr(d_ay[i]+my);
    
    for (int j=0;j<active_food;j++) {
        float dx=wd(d_ax[i],d_fx[j]),dy=wd(d_ay[i],d_fy[j]);
        if (dx*dx+dy*dy < d_grab*d_grab) {
            d_food[i]++;
            d_energy[i] += d_energy_gain;
            if (d_energy[i] > d_max_energy) d_energy[i] = d_max_energy;
            d_fx[j]=cr(&d_fseed[j])*WORLD;
            d_fy[j]=cr(&d_fseed[j])*WORLD;
            break;
        }
    }
}

void run(const char* label, int period, int feast, int with_energy, float cost, float gain, float maxe, int seed) {
    int _n=AGENTS, _nf=300;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(d_season_period, &period, sizeof(int));
    cudaMemcpyToSymbol(d_feast_fraction, &feast, sizeof(int));
    cudaMemcpyToSymbol(d_move_cost, &cost, sizeof(float));
    cudaMemcpyToSymbol(d_energy_gain, &gain, sizeof(float));
    cudaMemcpyToSymbol(d_max_energy, &maxe, sizeof(float));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(300+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) {
        step<<<blocks,BLOCK>>>(s);
        cudaDeviceSynchronize();
    }
    
    float h_food[AGENTS], h_energy[AGENTS];
    cudaMemcpyFromSymbol(h_food, d_food, sizeof(float)*AGENTS);
    cudaMemcpyFromSymbol(h_energy, d_energy, sizeof(float)*AGENTS);
    
    float total=0, alive=0;
    for (int i=0;i<AGENTS;i++) { total+=h_food[i]; if(h_energy[i]>0||!with_energy)alive++; }
    printf("  %-45s total=%.0f  alive=%.0f  per=%.1f\n",
           label, total, alive, alive>0?total/alive:0);
}

int main() {
    printf("=== Seasonal Bursts (Feast/Famine Cycles) ===\n\n");
    
    printf("--- No energy cost (baseline) ---\n");
    run("Always feast (control)", 1, 1, 0, 0, 0, 100, 42);
    run("50%% feast / 50%% famine (period=500)", 500, 250, 0, 0, 0, 100, 42);
    run("25%% feast / 75%% famine", 500, 125, 0, 0, 0, 100, 42);
    run("Short feast (period=100, 25)", 100, 25, 0, 0, 0, 100, 42);
    run("Long feast (period=1000, 500)", 1000, 500, 0, 0, 0, 100, 42);
    
    printf("\n--- With energy cost ---\n");
    run("Always feast, with energy", 1, 1, 1, 0.02f, 5.0f, 100, 42);
    run("50/50 seasonal, with energy", 500, 250, 1, 0.02f, 5.0f, 100, 42);
    run("25/75 seasonal, with energy", 500, 125, 1, 0.02f, 5.0f, 100, 42);
    run("High cost + 50/50 seasonal", 500, 250, 1, 0.05f, 10.0f, 100, 42);
    
    printf("\n--- Rapid cycles ---\n");
    run("Very short (period=20, feast=10)", 20, 10, 0, 0, 0, 100, 42);
    run("Ultra short (period=10, feast=5)", 10, 5, 0, 0, 0, 100, 42);
    
    return 0;
}
