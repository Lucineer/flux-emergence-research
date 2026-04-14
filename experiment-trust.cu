// experiment-trust.cu — Trust/reputation-based cooperation
// From SuperInstance/trust-agent: OCap tokens, trust engine
// Hypothesis: agents with high trust scores get better cooperation
// Test: trust as admission ticket to guild knowledge
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 256
#define FOOD 200
#define STEPS 3000
#define WORLD 1024
#define BLOCK 256
#define GUILD_TYPES 3

__device__ float ax[AGENTS], ay[AGENTS], a_food[AGENTS];
__device__ int a_seed[AGENTS], a_guild[AGENTS];
__device__ float a_trust[AGENTS]; // 0.0 to 1.0
__device__ float a_contrib[AGENTS]; // recent contributions
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD];
__device__ int d_n, d_nf, d_trust_mode; // 0=none, 1=trust-gated DCS, 2=reputation-weighted, 3=OCap tokens
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
    a_trust[i] = 0.5f; // start neutral
    a_contrib[i] = 0;
    if (i < d_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD;
        fy[i] = cr(&f_seed[i])*WORLD;
    }
    if (i < GUILD_TYPES) g_found[i] = 0;
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    
    float mx=0, my=0;
    int use_dcs = 0;
    
    if (d_trust_mode == 1) {
        // Trust-gated: only agents with trust > 0.3 can use DCS
        use_dcs = g_found[a_guild[i]] && a_trust[i] > 0.3f;
    } else if (d_trust_mode == 2) {
        // Reputation-weighted: DCS quality depends on trust
        use_dcs = g_found[a_guild[i]];
    } else if (d_trust_mode == 3) {
        // OCap tokens: agents spend trust to access DCS, gain trust by contributing
        use_dcs = g_found[a_guild[i]] && a_trust[i] > 0.1f;
    }
    
    if (use_dcs) {
        float dx=wd(g_best_x[a_guild[i]],ax[i]);
        float dy=wd(g_best_y[a_guild[i]],ay[i]);
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0){
            float spd = d_speed;
            if (d_trust_mode == 2) spd *= (0.5f + 0.5f * a_trust[i]); // trust-weighted speed
            mx=dx/d*spd; my=dy/d*spd;
        }
        if (d_trust_mode == 3) a_trust[i] = fmaxf(0, a_trust[i] - 0.01f); // spend trust
    } else {
        float best_d=1e9,bfx=0,bfy=0;
        for (int j=0;j<d_nf;j++) {
            float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
            float d=dx*dx+dy*dy;
            if(d<best_d){best_d=d;bfx=fx[j];bfy=fy[j];}
        }
        if (best_d < d_perc*d_perc) {
            float dx=wd(bfx,ax[i]),dy=wd(bfy,ay[i]);
            float d=sqrtf(best_d); if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
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
            // Update guild knowledge
            g_best_x[a_guild[i]]=fx[j];
            g_best_y[a_guild[i]]=fy[j];
            g_found[a_guild[i]]=1;
            // Build trust by contributing
            if (d_trust_mode >= 1) {
                a_trust[i] = fminf(1.0f, a_trust[i] + 0.05f);
                a_contrib[i] += 1.0f;
            }
            fx[j]=cr(&f_seed[j])*WORLD;
            fy[j]=cr(&f_seed[j])*WORLD;
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
    cudaMemcpyToSymbol(d_trust_mode, &mode, sizeof(int));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[AGENTS], h_trust[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*AGENTS);
    cudaMemcpyFromSymbol(h_trust, a_trust, sizeof(float)*AGENTS);
    
    float total=0, high_trust_food=0, high_trust_count=0;
    for (int i=0;i<AGENTS;i++) {
        total+=h_food[i];
        if (h_trust[i]>0.5) { high_trust_food+=h_food[i]; high_trust_count++; }
    }
    printf("  %-45s total=%.0f  per=%.1f  hi_trust=%.0f/agent (%d agents)\n",
           label, total, total/AGENTS, high_trust_count>0?high_trust_food/high_trust_count:0, high_trust_count);
}

int main() {
    printf("=== Trust-Based Cooperation (SuperInstance/trust-agent) ===\n\n");
    
    printf("--- Baseline ---\n");
    run("No trust, no DCS (control)", 0, 42);
    
    printf("\n--- Trust Modes ---\n");
    run("Trust-gated DCS (trust>0.3 to access)", 1, 42);
    run("Reputation-weighted DCS speed", 2, 42);
    run("OCap tokens (spend trust to access)", 3, 42);
    
    printf("\n--- Scarcity variant (100 food) ---\n");
    int nf100=100; cudaMemcpyToSymbol(d_nf, &nf100, sizeof(int));
    run("No trust, scarcity", 0, 42);
    run("Trust-gated, scarcity", 1, 42);
    run("OCap tokens, scarcity", 3, 42);
    
    return 0;
}
