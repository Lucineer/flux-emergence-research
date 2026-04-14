// experiment-energy-cost.cu — Moving costs energy, agents have energy budget
// Hypothesis: energy constraints change optimal movement strategy
// Counter: grab range dominates so much that energy doesn't matter
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 512
#define FOOD 200
#define STEPS 5000
#define WORLD 1024
#define BLOCK 256

__device__ float d_ax[AGENTS], d_ay[AGENTS], d_food[AGENTS], d_energy[AGENTS];
__device__ int d_seed[AGENTS];
__device__ float d_fx[FOOD], d_fy[FOOD];
__device__ int d_fseed[FOOD];
__device__ int d_n, d_nf;
__device__ float d_grab, d_speed, d_perc, d_move_cost, d_energy_gain, d_max_energy;

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
    if (i < d_nf) {
        d_fseed[i] = seed+50000+i*997;
        d_fx[i] = cr(&d_fseed[i])*WORLD;
        d_fy[i] = cr(&d_fseed[i])*WORLD;
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    if (d_energy[i] <= 0) return; // dead agent
    
    float best_d=1e9, bfx=0, bfy=0;
    for (int j=0;j<d_nf;j++) {
        float dx=wd(d_ax[i],d_fx[j]), dy=wd(d_ay[i],d_fy[j]);
        float d=dx*dx+dy*dy;
        if (d<best_d) { best_d=d; bfx=d_fx[j]; bfy=d_fy[j]; }
    }
    
    float mx=0,my=0;
    if (best_d < d_perc*d_perc) {
        float dx=wd(bfx,d_ax[i]), dy=wd(bfy,d_ay[i]);
        float d=sqrtf(best_d);
        if(d>0){mx=dx/d*d_speed; my=dy/d*d_speed;}
    } else {
        float ang=cr(&d_seed[i])*6.2832f;
        mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
    }
    
    // Apply movement cost
    float move_dist = sqrtf(mx*mx+my*my);
    d_energy[i] -= move_dist * d_move_cost;
    if (d_energy[i] <= 0) return;
    
    d_ax[i]=wr(d_ax[i]+mx); d_ay[i]=wr(d_ay[i]+my);
    
    for (int j=0;j<d_nf;j++) {
        float dx=wd(d_ax[i],d_fx[j]), dy=wd(d_ay[i],d_fy[j]);
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

void run(const char* label, float move_cost, float energy_gain, float max_energy, int seed) {
    int _n=AGENTS, _nf=FOOD;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(d_move_cost, &move_cost, sizeof(float));
    cudaMemcpyToSymbol(d_energy_gain, &energy_gain, sizeof(float));
    cudaMemcpyToSymbol(d_max_energy, &max_energy, sizeof(float));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[AGENTS], h_energy[AGENTS];
    cudaMemcpyFromSymbol(h_food, d_food, sizeof(float)*AGENTS);
    cudaMemcpyFromSymbol(h_energy, d_energy, sizeof(float)*AGENTS);
    
    float total=0, alive=0, avg_energy=0;
    for (int i=0;i<AGENTS;i++) {
        total += h_food[i];
        if (h_energy[i] > 0) { alive++; avg_energy += h_energy[i]; }
    }
    printf("  %-40s total=%.0f  alive=%.0f  avg_energy=%.1f  per_alive=%.1f\n",
           label, total, alive, alive>0?avg_energy/alive:0, alive>0?total/alive:0);
}

int main() {
    printf("=== Energy Costs — Moving Costs Energy ===\n");
    printf("Agents start with energy, lose energy when moving, gain energy from food\n\n");
    
    printf("--- No energy cost (control) ---\n");
    run("No cost, gain=0", 0.0f, 0.0f, 1000.0f, 42);
    
    printf("\n--- Low cost (0.01 per unit moved) ---\n");
    run("Cost=0.01, gain=5", 0.01f, 5.0f, 100.0f, 42);
    run("Cost=0.01, gain=10", 0.01f, 10.0f, 100.0f, 42);
    run("Cost=0.01, gain=20", 0.01f, 20.0f, 100.0f, 42);
    
    printf("\n--- Moderate cost (0.05) ---\n");
    run("Cost=0.05, gain=5", 0.05f, 5.0f, 100.0f, 42);
    run("Cost=0.05, gain=10", 0.05f, 10.0f, 100.0f, 42);
    run("Cost=0.05, gain=20", 0.05f, 20.0f, 100.0f, 42);
    
    printf("\n--- High cost (0.1) — movement expensive ---\n");
    run("Cost=0.1, gain=5", 0.1f, 5.0f, 100.0f, 42);
    run("Cost=0.1, gain=10", 0.1f, 10.0f, 100.0f, 42);
    run("Cost=0.1, gain=20", 0.1f, 20.0f, 100.0f, 42);
    
    printf("\n--- Extreme cost (0.5) — movement very expensive ---\n");
    run("Cost=0.5, gain=20", 0.5f, 20.0f, 100.0f, 42);
    run("Cost=0.5, gain=50", 0.5f, 50.0f, 100.0f, 42);
    
    printf("\n=== Analysis ===\n");
    printf("If alive agents < total: energy constraints kill agents\n");
    printf("If per_alive similar to control: energy doesn't matter, survivors adapt\n");
    printf("If total drops sharply: energy constraints dominate strategy\n");
    
    return 0;
}
