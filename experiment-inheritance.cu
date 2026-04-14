// experiment-inheritance.cu — Death with cultural inheritance
// Dead agents respawn with the best surviving agent's memory
// Hypothesis: cultural transmission makes convergence work across generations
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 512
#define FOOD 200
#define STEPS 5000
#define WORLD 1024
#define BLOCK 256
#define MAX_MEM 8

__device__ float d_ax[AGENTS], d_ay[AGENTS], d_food[AGENTS], d_energy[AGENTS];
__device__ int d_seed[AGENTS];
__device__ float d_mem_x[AGENTS*MAX_MEM], d_mem_y[AGENTS*MAX_MEM];
__device__ int d_mem_idx[AGENTS], d_mem_count[AGENTS];
__device__ float d_fx[FOOD], d_fy[FOOD];
__device__ int d_fseed[FOOD];
__device__ int d_n, d_nf;
__device__ float d_grab, d_speed, d_perc, d_move_cost, d_energy_gain, d_max_energy;
__device__ int d_inherit; // 0=none, 1=best agent, 2=random survivor, 3=converge to best

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
    d_mem_idx[i] = 0;
    d_mem_count[i] = 0;
    for (int m=0; m<MAX_MEM; m++) { d_mem_x[i*MAX_MEM+m]=0; d_mem_y[i*MAX_MEM+m]=0; }
    if (i < d_nf) {
        d_fseed[i] = seed+50000+i*997;
        d_fx[i] = cr(&d_fseed[i])*WORLD;
        d_fy[i] = cr(&d_fseed[i])*WORLD;
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    if (d_energy[i] <= 0) return;
    
    // Predict from memory
    float pred_x=-1, pred_y=-1;
    int has_pred = 0;
    if (d_mem_count[i] >= 2) {
        int i1 = (d_mem_idx[i]-1+MAX_MEM)%MAX_MEM;
        int i2 = (d_mem_idx[i]-2+MAX_MEM)%MAX_MEM;
        pred_x = wr(d_mem_x[i*MAX_MEM+i1] + (d_mem_x[i*MAX_MEM+i1] - d_mem_x[i*MAX_MEM+i2]));
        pred_y = wr(d_mem_y[i*MAX_MEM+i1] + (d_mem_y[i*MAX_MEM+i1] - d_mem_y[i*MAX_MEM+i2]));
        has_pred = 1;
    } else if (d_mem_count[i] >= 1) {
        int i1 = (d_mem_idx[i]-1+MAX_MEM)%MAX_MEM;
        pred_x = d_mem_x[i*MAX_MEM+i1];
        pred_y = d_mem_y[i*MAX_MEM+i1];
        has_pred = 1;
    }
    
    // Find nearest food
    float best_d=1e9, bfx=0, bfy=0;
    for (int j=0;j<d_nf;j++) {
        float dx=wd(d_ax[i],d_fx[j]), dy=wd(d_ay[i],d_fy[j]);
        float d=dx*dx+dy*dy;
        if (d<best_d) { best_d=d; bfx=d_fx[j]; bfy=d_fy[j]; }
    }
    
    float mx=0,my=0;
    if (best_d < d_perc*d_perc) {
        float dx=wd(bfx,d_ax[i]),dy=wd(bfy,d_ay[i]);
        float d=sqrtf(best_d); if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
        // Store memory
        d_mem_x[i*MAX_MEM+d_mem_idx[i]] = bfx;
        d_mem_y[i*MAX_MEM+d_mem_idx[i]] = bfy;
        d_mem_idx[i] = (d_mem_idx[i]+1)%MAX_MEM;
        if (d_mem_count[i] < MAX_MEM) d_mem_count[i]++;
    } else if (has_pred) {
        float dx=wd(pred_x,d_ax[i]),dy=wd(pred_y,d_ay[i]);
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
    } else {
        float ang=cr(&d_seed[i])*6.2832f;
        mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
    }
    
    d_energy[i] -= sqrtf(mx*mx+my*my) * d_move_cost;
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

__global__ void respawn_dead(int step) {
    // Find best living agent
    int best_idx = -1;
    float best_food = -1;
    for (int i=0; i<d_n; i++) {
        if (d_energy[i] > 0 && d_food[i] > best_food) {
            best_food = d_food[i];
            best_idx = i;
        }
    }
    if (best_idx < 0) return;
    
    for (int i=0; i<d_n; i++) {
        if (d_energy[i] > 0) continue;
        if (d_inherit == 0) {
            // No inheritance — fresh respawn
            d_energy[i] = d_max_energy;
            d_ax[i] = cr(&d_seed[i])*WORLD;
            d_ay[i] = cr(&d_seed[i])*WORLD;
            d_mem_count[i] = 0;
        } else if (d_inherit == 1) {
            // Inherit best agent's memory
            d_energy[i] = d_max_energy;
            d_ax[i] = d_ax[best_idx] + (cr(&d_seed[i])-0.5f)*50;
            d_ay[i] = d_ay[best_idx] + (cr(&d_seed[i])-0.5f)*50;
            d_ax[i] = wr(d_ax[i]); d_ay[i] = wr(d_ay[i]);
            for (int m=0; m<MAX_MEM; m++) {
                d_mem_x[i*MAX_MEM+m] = d_mem_x[best_idx*MAX_MEM+m];
                d_mem_y[i*MAX_MEM+m] = d_mem_y[best_idx*MAX_MEM+m];
            }
            d_mem_idx[i] = d_mem_idx[best_idx];
            d_mem_count[i] = d_mem_count[best_idx];
        } else if (d_inherit == 2) {
            // Random survivor's memory
            int tries = 0, src = -1;
            while (tries < 20) {
                src = (int)(cr(&d_seed[i]) * d_n);
                if (src >= 0 && src < d_n && d_energy[src] > 0) break;
                tries++;
            }
            if (src < 0 || d_energy[src] <= 0) src = best_idx;
            d_energy[i] = d_max_energy;
            d_ax[i] = d_ax[src] + (cr(&d_seed[i])-0.5f)*50;
            d_ay[i] = d_ay[src] + (cr(&d_seed[i])-0.5f)*50;
            d_ax[i] = wr(d_ax[i]); d_ay[i] = wr(d_ay[i]);
            for (int m=0; m<MAX_MEM; m++) {
                d_mem_x[i*MAX_MEM+m] = d_mem_x[src*MAX_MEM+m];
                d_mem_y[i*MAX_MEM+m] = d_mem_y[src*MAX_MEM+m];
            }
            d_mem_idx[i] = d_mem_idx[src];
            d_mem_count[i] = d_mem_count[src];
        } else if (d_inherit == 3) {
            // Converge: move toward best agent
            d_energy[i] = d_max_energy * 0.5f; // partial energy
            float dx=wd(d_ax[best_idx],d_ax[i]), dy=wd(d_ay[best_idx],d_ay[i]);
            d_ax[i] = wr(d_ax[i]+dx*0.7f);
            d_ay[i] = wr(d_ay[i]+dy*0.7f);
            d_mem_count[i] = 0;
        }
    }
}

void run(const char* label, int inherit, float cost, float gain, float max_e, int seed) {
    int _n=AGENTS, _nf=FOOD;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(d_move_cost, &cost, sizeof(float));
    cudaMemcpyToSymbol(d_energy_gain, &gain, sizeof(float));
    cudaMemcpyToSymbol(d_max_energy, &max_e, sizeof(float));
    cudaMemcpyToSymbol(d_inherit, &inherit, sizeof(int));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    
    int respawn_interval = 100;
    for (int s=0; s<STEPS; s++) {
        step<<<blocks,BLOCK>>>();
        if (inherit > 0 && s % respawn_interval == 0 && s > 0)
            respawn_dead<<<1,1>>>(s);
        cudaDeviceSynchronize();
    }
    
    float h_food[AGENTS], h_energy[AGENTS];
    cudaMemcpyFromSymbol(h_food, d_food, sizeof(float)*AGENTS);
    cudaMemcpyFromSymbol(h_energy, d_energy, sizeof(float)*AGENTS);
    
    float total=0, alive=0;
    for (int i=0;i<AGENTS;i++) { total+=h_food[i]; if(h_energy[i]>0)alive++; }
    printf("  %-40s total=%.0f  alive=%.0f/512  per=%.1f\n",
           label, total, alive, alive>0?total/alive:0);
}

int main() {
    printf("=== Death with Cultural Inheritance ===\n\n");
    
    // First: find the energy cost that kills some but not all
    printf("--- Baseline: energy without inheritance ---\n");
    run("No inherit, cost=0.03, gain=5", 0, 0.03f, 5.0f, 100.0f, 42);
    run("No inherit, cost=0.05, gain=10", 0, 0.05f, 10.0f, 100.0f, 42);
    run("No inherit, cost=0.05, gain=15", 0, 0.05f, 15.0f, 100.0f, 42);
    run("No inherit, cost=0.08, gain=15", 0, 0.08f, 15.0f, 100.0f, 42);
    
    // Use cost=0.05, gain=10 as the sweet spot (kills some, survivors exist)
    printf("\n--- Inheritance modes (cost=0.05, gain=10) ---\n");
    run("No inherit (control)", 0, 0.05f, 10.0f, 100.0f, 42);
    run("Inherit best agent's memory", 1, 0.05f, 10.0f, 100.0f, 42);
    run("Inherit random survivor", 2, 0.05f, 10.0f, 100.0f, 42);
    run("Converge to best position", 3, 0.05f, 10.0f, 100.0f, 42);
    
    printf("\n--- Higher death rate (cost=0.08, gain=15) ---\n");
    run("No inherit, high death", 0, 0.08f, 15.0f, 100.0f, 42);
    run("Inherit best, high death", 1, 0.08f, 15.0f, 100.0f, 42);
    run("Converge, high death", 3, 0.08f, 15.0f, 100.0f, 42);
    
    printf("\n--- No death baseline (cost=0.01) ---\n");
    run("No death, no inherit", 0, 0.01f, 5.0f, 100.0f, 42);
    run("No death, inherit best", 1, 0.01f, 5.0f, 100.0f, 42);
    
    return 0;
}
