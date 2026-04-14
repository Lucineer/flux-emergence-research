// experiment-dcs-memory.cu — DCS protocol + spatial memory prediction
// Does structured communication + individual prediction stack or conflict?
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
#define MAX_MEM 8

// Agent state
__device__ float ax[AGENTS], ay[AGENTS], a_food[AGENTS];
__device__ int a_seed[AGENTS], a_guild[AGENTS];
__device__ float a_mem_x[AGENTS*MAX_MEM], a_mem_y[AGENTS*MAX_MEM];
__device__ int a_mem_idx[AGENTS], a_mem_count[AGENTS];

// Food with migration
__device__ float fx[FOOD], fy[FOOD], fx_vel[FOOD], fy_vel[FOOD];
__device__ int f_seed[FOOD];

// Config
__device__ int cfg_n, cfg_nf, cfg_dcs, cfg_memory, cfg_migrate;
__device__ float cfg_grab, cfg_speed, cfg_perc, cfg_migrate_speed;

// DCS: guild knowledge table
__device__ float guild_best_x[GUILD_TYPES], guild_best_y[GUILD_TYPES];
__device__ float guild_best_food[GUILD_TYPES];
__device__ int guild_found[GUILD_TYPES]; // has guild found food recently?

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
    a_guild[i] = i % GUILD_TYPES;
    a_mem_idx[i] = 0;
    a_mem_count[i] = 0;
    for (int m=0;m<MAX_MEM;m++) { a_mem_x[i*MAX_MEM+m]=0; a_mem_y[i*MAX_MEM+m]=0; }
    if (i < cfg_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD;
        fy[i] = cr(&f_seed[i])*WORLD;
        float ang = cr(&f_seed[i])*6.2832f;
        fx_vel[i] = cosf(ang)*cfg_migrate_speed;
        fy_vel[i] = sinf(ang)*cfg_migrate_speed;
    }
    if (i < GUILD_TYPES) {
        guild_best_food[i] = 0;
        guild_found[i] = 0;
    }
}

__global__ void do_migrate() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= cfg_nf || !cfg_migrate) return;
    fx[i] = wr(fx[i] + fx_vel[i]);
    fy[i] = wr(fy[i] + fy_vel[i]);
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= cfg_n) return;
    
    float mx=0, my=0;
    
    // Priority 1: DCS — check guild knowledge
    if (cfg_dcs && guild_found[a_guild[i]]) {
        float dx=wd(guild_best_x[a_guild[i]],ax[i]);
        float dy=wd(guild_best_y[a_guild[i]],ay[i]);
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0){mx=dx/d*cfg_speed;my=dy/d*cfg_speed;}
    }
    // Priority 2: Memory prediction
    else if (cfg_memory && a_mem_count[i] >= 2) {
        int i1=(a_mem_idx[i]-1+MAX_MEM)%MAX_MEM;
        int i2=(a_mem_idx[i]-2+MAX_MEM)%MAX_MEM;
        float px=wr(a_mem_x[i*MAX_MEM+i1]+(a_mem_x[i*MAX_MEM+i1]-a_mem_x[i*MAX_MEM+i2]));
        float py=wr(a_mem_y[i*MAX_MEM+i1]+(a_mem_y[i*MAX_MEM+i1]-a_mem_y[i*MAX_MEM+i2]));
        float dx=wd(px,ax[i]),dy=wd(py,ay[i]);
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0){mx=dx/d*cfg_speed;my=dy/d*cfg_speed;}
    }
    // Priority 3: Perception
    else {
        float best_d=1e9, bfx=0, bfy=0;
        for (int j=0;j<cfg_nf;j++) {
            float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
            float d=dx*dx+dy*dy;
            if(d<best_d){best_d=d;bfx=fx[j];bfy=fy[j];}
        }
        if (best_d < cfg_perc*cfg_perc) {
            float dx=wd(bfx,ax[i]),dy=wd(bfy,ay[i]);
            float d=sqrtf(best_d);
            if(d>0){mx=dx/d*cfg_speed;my=dy/d*cfg_speed;}
            if (cfg_memory) {
                a_mem_x[i*MAX_MEM+a_mem_idx[i]]=bfx;
                a_mem_y[i*MAX_MEM+a_mem_idx[i]]=bfy;
                a_mem_idx[i]=(a_mem_idx[i]+1)%MAX_MEM;
                if(a_mem_count[i]<MAX_MEM)a_mem_count[i]++;
            }
        } else {
            float ang=cr(&a_seed[i])*6.2832f;
            mx=cosf(ang)*cfg_speed; my=sinf(ang)*cfg_speed;
        }
    }
    
    ax[i]=wr(ax[i]+mx); ay[i]=wr(ay[i]+my);
    
    for (int j=0;j<cfg_nf;j++) {
        float dx=wd(ax[i],fx[j]),dy=wd(ay[i],fy[j]);
        if(dx*dx+dy*dy < cfg_grab*cfg_grab) {
            a_food[i]++;
            // Update guild knowledge
            if (cfg_dcs) {
                guild_best_x[a_guild[i]] = fx[j]; // approximate
                guild_best_y[a_guild[i]] = fy[j];
                guild_best_food[a_guild[i]] += 1.0f;
                guild_found[a_guild[i]] = 1;
            }
            // Respawn food
            fx[j]=cr(&f_seed[j])*WORLD;
            fy[j]=cr(&f_seed[j])*WORLD;
            float ang=cr(&f_seed[j])*6.2832f;
            fx_vel[j]=cosf(ang)*cfg_migrate_speed;
            fy_vel[j]=sinf(ang)*cfg_migrate_speed;
            break;
        }
    }
}

void run(const char* label, int dcs, int memory, int migrate, float mspeed, int seed) {
    int _n=AGENTS, _nf=FOOD;
    float _g=15.0f, _sp=3.0f, _p=50.0f;
    cudaMemcpyToSymbol(cfg_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(cfg_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(cfg_grab, &_g, sizeof(float));
    cudaMemcpyToSymbol(cfg_speed, &_sp, sizeof(float));
    cudaMemcpyToSymbol(cfg_perc, &_p, sizeof(float));
    cudaMemcpyToSymbol(cfg_dcs, &dcs, sizeof(int));
    cudaMemcpyToSymbol(cfg_memory, &memory, sizeof(int));
    cudaMemcpyToSymbol(cfg_migrate, &migrate, sizeof(int));
    cudaMemcpyToSymbol(cfg_migrate_speed, &mspeed, sizeof(float));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) {
        do_migrate<<<fblocks,BLOCK>>>();
        step<<<blocks,BLOCK>>>();
        cudaDeviceSynchronize();
    }
    
    float h_food[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*AGENTS);
    float total=0;
    for (int i=0;i<AGENTS;i++) total+=h_food[i];
    printf("  %-50s total=%.0f  per=%.1f\n", label, total, total/AGENTS);
}

int main() {
    printf("=== DCS + Spatial Memory ===\n\n");
    
    printf("--- Static food ---\n");
    run("Control (no DCS, no memory)", 0, 0, 0, 0, 42);
    run("DCS only", 1, 0, 0, 0, 42);
    run("Memory only", 0, 1, 0, 0, 42);
    run("DCS + Memory", 1, 1, 0, 0, 42);
    
    printf("\n--- Migrating food (speed=1.0) ---\n");
    run("Control, migrate", 0, 0, 1, 1.0f, 42);
    run("DCS only, migrate", 1, 0, 1, 1.0f, 42);
    run("Memory only, migrate", 0, 1, 1, 1.0f, 42);
    run("DCS + Memory, migrate", 1, 1, 1, 1.0f, 42);
    
    printf("\n--- Fast migration (speed=3.0) ---\n");
    run("Control, fast migrate", 0, 0, 1, 3.0f, 42);
    run("DCS only, fast migrate", 1, 0, 1, 3.0f, 42);
    run("Memory only, fast migrate", 0, 1, 1, 3.0f, 42);
    run("DCS + Memory, fast migrate", 1, 1, 1, 3.0f, 42);
    
    printf("\n=== Key Question ===\n");
    printf("If DCS+Memory > DCS alone: mechanisms STACK (multiplicative)\n");
    printf("If DCS+Memory ≈ DCS alone: DCS dominates, memory irrelevant under DCS\n");
    printf("If DCS+Memory < DCS alone: mechanisms INTERFERE (conflict)\n");
    
    return 0;
}
