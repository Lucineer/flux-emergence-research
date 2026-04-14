// experiment-gradient.cu — Food distribution gradient
// Half the world is rich (high food density), other half is poor
// Hypothesis: agents that stay in the rich half outperform; 
// does any agent naturally find and stay in the rich zone?
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 512
#define FOOD 300
#define STEPS 5000
#define WORLD 1024
#define BLOCK 256
#define ZONE 512 // rich zone = x < ZONE

__device__ float d_ax[AGENTS], d_ay[AGENTS], d_food[AGENTS];
__device__ int d_seed[AGENTS];
__device__ float d_fx[FOOD], d_fy[FOOD];
__device__ int d_fseed[FOOD];
__device__ int d_n, d_nf, d_steps;
__device__ float d_grab, d_speed, d_perc;
__device__ float d_rich_prob; // probability of food spawning in rich zone

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a,float b) { float d=fabsf(a-b); return fminf(d,WORLD-d); }
__device__ float wr(float v) { float r=fmodf(v,WORLD); return r<0?r+WORLD:r; }

__global__ void init(int seed, float rich_prob) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    d_seed[i] = seed+i*137;
    d_ax[i] = cr(&d_seed[i])*WORLD;
    d_ay[i] = cr(&d_seed[i])*WORLD;
    d_food[i] = 0;
    if (i < d_nf) {
        d_fseed[i] = seed+50000+i*997;
        // Gradient spawn: rich_prob chance in left half, (1-rich_prob) in right
        float fx = cr(&d_fseed[i])*WORLD;
        if (fx > ZONE && cr(&d_fseed[i]) > rich_prob) {
            fx = cr(&d_fseed[i])*ZONE; // force into rich zone
        }
        d_fx[i] = fx;
        d_fy[i] = cr(&d_fseed[i])*WORLD;
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    
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
    } else {
        float ang=cr(&d_seed[i])*6.2832f;
        mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
    }
    d_ax[i]=wr(d_ax[i]+mx); d_ay[i]=wr(d_ay[i]+my);
    
    for (int j=0;j<d_nf;j++) {
        float dx=wd(d_ax[i],d_fx[j]), dy=wd(d_ay[i],d_fy[j]);
        if (dx*dx+dy*dy < d_grab*d_grab) {
            d_food[i]++;
            // Respawn with gradient
            float fx = cr(&d_fseed[j])*WORLD;
            if (fx > ZONE && cr(&d_fseed[j]) > d_rich_prob) {
                fx = cr(&d_fseed[j])*ZONE;
            }
            d_fx[j] = fx;
            d_fy[j] = cr(&d_fseed[j])*WORLD;
            break;
        }
    }
}

void run(const char* label, float rich_prob, int seed) {
    int _n=AGENTS, _nf=FOOD, _st=STEPS;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(d_rich_prob, &rich_prob, sizeof(float));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed, rich_prob);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[AGENTS], h_ax[AGENTS];
    cudaMemcpyFromSymbol(h_food, d_food, sizeof(float)*AGENTS);
    cudaMemcpyFromSymbol(h_ax, d_ax, sizeof(float)*AGENTS);
    
    float total=0, rich_agents=0, rich_food=0, poor_agents=0, poor_food=0;
    for (int i=0;i<AGENTS;i++) {
        total += h_food[i];
        if (h_ax[i] < ZONE) { rich_agents++; rich_food += h_food[i]; }
        else { poor_agents++; poor_food += h_food[i]; }
    }
    printf("  %-35s total=%.0f  rich=%.1f(%d)  poor=%.1f(%d)  ratio=%.2f\n",
           label, total,
           rich_agents>0?rich_food/rich_agents:0, (int)rich_agents,
           poor_agents>0?poor_food/poor_agents:0, (int)poor_agents,
           rich_agents>0 && poor_agents>0 ? (rich_food/rich_agents)/(poor_food/poor_agents) : 0);
}

int main() {
    printf("=== Environmental Gradient ===\n");
    printf("Left half = rich zone (food spawns here with higher probability)\n\n");
    
    printf("--- Gradient intensity sweep ---\n");
    run("Uniform (control, 50%%)", 0.5f, 42);
    run("Mild gradient (60%% rich)", 0.6f, 42);
    run("Moderate gradient (70%% rich)", 0.7f, 42);
    run("Strong gradient (80%% rich)", 0.8f, 42);
    run("Extreme gradient (90%% rich)", 0.9f, 42);
    run("Nearly all in rich (95%%)", 0.95f, 42);
    
    printf("\n--- Interpretation ---\n");
    printf("If rich agents >> poor agents: agents self-select into rich zone\n");
    printf("If ratio grows with gradient: emergence of spatial specialization\n");
    printf("If total fitness drops: gradient creates unequal access (scarcity for poor half)\n");
    
    return 0;
}
