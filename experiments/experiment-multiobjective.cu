// experiment-multiobjective.cu
// Multi-Objective: agents balance collection vs exploration vs rest
//
// Real agents don't just collect. They explore, rest, and conserve energy.
// Hypothesis: agents that allocate time across objectives outperform
// single-minded collectors because they discover new food patches.
//
// Three objectives with tradeoffs:
// - COLLECT: grab nearby food (immediate reward)
// - EXPLORE: move toward unexplored areas (future reward)
// - REST: stay still, recover energy (enables future bursts)
//
// Agent state: energy (0-100), collection rate affects energy drain

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID 256
#define NA 1024
#define NR 300
#define STEPS 4000
#define TRIALS 5

struct A {
    float x, y;
    float energy;
    int col;
    int explored; // cells visited
    float spd;
    int state; // 0=collect, 1=explore, 2=rest
    int state_timer;
};

struct R {
    float x, y;
    int active;
};

__device__ int d_tot, d_explored;
__device__ float rng(unsigned int *s) { *s = *s * 1103515245u + 12345u; return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu; }

__global__ void reset() { if (threadIdx.x == 0 && blockIdx.x == 0) { d_tot = 0; d_explored = 0; } }

__global__ void initA(A *a, int n, unsigned int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    unsigned int r = s + i * 7919;
    a[i].x = rng(&r) * GRID; a[i].y = rng(&r) * GRID;
    a[i].energy = 50.0f + rng(&r) * 50.0f;
    a[i].col = 0; a[i].explored = 0; a[i].spd = 3.0f;
    a[i].state = 0; a[i].state_timer = 0;
}

__global__ void initR(R *r, int n, unsigned int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    unsigned int r2 = s + i * 65537;
    r[i].x = rng(&r2) * GRID; r[i].y = rng(&r2) * GRID; r[i].active = 1;
}

// MODE 0: Collect only (control)
__global__ void step_collect(A *a, int na, R *r, int nr, unsigned int s, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= na) return;
    unsigned int r2 = s + i * 131 + step * 997;
    
    a[i].x = fmodf(a[i].x + (rng(&r2) - 0.5f) * a[i].spd * 2 + GRID, GRID);
    a[i].y = fmodf(a[i].y + (rng(&r2) - 0.5f) * a[i].spd * 2 + GRID, GRID);
    
    float g2 = 9.0f;
    for (int j = 0; j < nr; j++) {
        if (!r[j].active) continue;
        float dx = a[i].x - r[j].x, dy = a[i].y - r[j].y;
        if (dx > GRID*0.5f) dx -= GRID; if (dx < -GRID*0.5f) dx += GRID;
        if (dy > GRID*0.5f) dy -= GRID; if (dy < -GRID*0.5f) dy += GRID;
        if (dx*dx + dy*dy < g2) { r[j].active = 0; a[i].col++; atomicAdd(&d_tot, 1); break; }
    }
}

// MODE 1: Multi-objective with fixed allocation (e.g., 70% collect, 20% explore, 10% rest)
__global__ void step_multi_fixed(A *a, int na, R *r, int nr, unsigned int s, int step,
                                   float p_collect, float p_explore, float energy_cost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= na) return;
    unsigned int r2 = s + i * 131 + step * 997;
    
    float roll = rng(&r2);
    
    if (a[i].energy < 10.0f) {
        // Force rest when low energy
        a[i].state = 2;
    } else if (roll < p_collect) {
        a[i].state = 0;
    } else if (roll < p_collect + p_explore) {
        a[i].state = 1;
    } else {
        a[i].state = 2;
    }
    
    switch (a[i].state) {
        case 0: // COLLECT
            a[i].x = fmodf(a[i].x + (rng(&r2) - 0.5f) * a[i].spd * 2 + GRID, GRID);
            a[i].y = fmodf(a[i].y + (rng(&r2) - 0.5f) * a[i].spd * 2 + GRID, GRID);
            a[i].energy -= energy_cost;
            {
                float g2 = 9.0f;
                for (int j = 0; j < nr; j++) {
                    if (!r[j].active) continue;
                    float dx = a[i].x - r[j].x, dy = a[i].y - r[j].y;
                    if (dx > GRID*0.5f) dx -= GRID; if (dx < -GRID*0.5f) dx += GRID;
                    if (dy > GRID*0.5f) dy -= GRID; if (dy < -GRID*0.5f) dy += GRID;
                    if (dx*dx + dy*dy < g2) { r[j].active = 0; a[i].col++; a[i].energy += 10.0f; atomicAdd(&d_tot, 1); break; }
                }
            }
            break;
        case 1: // EXPLORE (move in straight line, cover new ground)
            a[i].x = fmodf(a[i].x + cosf(i * 0.1f + step * 0.02f) * a[i].spd * 1.5f + GRID, GRID);
            a[i].y = fmodf(a[i].y + sinf(i * 0.1f + step * 0.02f) * a[i].spd * 1.5f + GRID, GRID);
            a[i].energy -= energy_cost * 0.5f;
            atomicAdd(&d_explored, 1);
            // Can still grab food while exploring
            {
                float g2 = 9.0f;
                for (int j = 0; j < nr; j++) {
                    if (!r[j].active) continue;
                    float dx = a[i].x - r[j].x, dy = a[i].y - r[j].y;
                    if (dx > GRID*0.5f) dx -= GRID; if (dx < -GRID*0.5f) dx += GRID;
                    if (dy > GRID*0.5f) dy -= GRID; if (dy < -GRID*0.5f) dy += GRID;
                    if (dx*dx + dy*dy < g2) { r[j].active = 0; a[i].col++; a[i].energy += 10.0f; atomicAdd(&d_tot, 1); break; }
                }
            }
            break;
        case 2: // REST (stay still, recover energy)
            a[i].energy = fminf(100.0f, a[i].energy + 2.0f);
            break;
    }
    
    a[i].energy = fmaxf(0.0f, fminf(100.0f, a[i].energy));
}

// MODE 2: Adaptive multi-objective (agent adjusts based on energy level)
__global__ void step_multi_adaptive(A *a, int na, R *r, int nr, unsigned int s, int step,
                                     float energy_cost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= na) return;
    unsigned int r2 = s + i * 131 + step * 997;
    
    // Adaptive allocation based on energy
    float p_collect, p_explore;
    if (a[i].energy > 70.0f) {
        p_collect = 0.6f; p_explore = 0.35f; // lots of energy: explore more
    } else if (a[i].energy > 30.0f) {
        p_collect = 0.8f; p_explore = 0.15f; // medium: mostly collect
    } else {
        p_collect = 0.3f; p_explore = 0.0f;  // low: rest mostly
    }
    
    float roll = rng(&r2);
    if (roll < p_collect) a[i].state = 0;
    else if (roll < p_collect + p_explore) a[i].state = 1;
    else a[i].state = 2;
    
    switch (a[i].state) {
        case 0:
            a[i].x = fmodf(a[i].x + (rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
            a[i].y = fmodf(a[i].y + (rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
            a[i].energy -= energy_cost;
            {float g2=9.0f;for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
            if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
            if(dx*dx+dy*dy<g2){r[j].active=0;a[i].col++;a[i].energy+=10.0f;atomicAdd(&d_tot,1);break;}}}
            break;
        case 1:
            a[i].x=fmodf(a[i].x+cosf(i*0.1f+step*0.02f)*a[i].spd*1.5f+GRID,GRID);
            a[i].y=fmodf(a[i].y+sinf(i*0.1f+step*0.02f)*a[i].spd*1.5f+GRID,GRID);
            a[i].energy-=energy_cost*0.5f;atomicAdd(&d_explored,1);
            {float g2=9.0f;for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
            if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
            if(dx*dx+dy*dy<g2){r[j].active=0;a[i].col++;a[i].energy+=10.0f;atomicAdd(&d_tot,1);break;}}}
            break;
        case 2:
            a[i].energy=fminf(100.0f,a[i].energy+2.0f);break;
    }
    a[i].energy=fmaxf(0.0f,fminf(100.0f,a[i].energy));
}

__global__ void respawn(R *r, int n, unsigned int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || r[i].active) return;
    unsigned int r2 = s + i * 3571;
    r[i].x = rng(&r2) * GRID; r[i].y = rng(&r2) * GRID; r[i].active = 1;
}

int main() {
    printf("=== Multi-Objective Agents ===\n");
    printf("Agents:%d Resources:%d Steps:%d Trials:%d\n\n", NA, NR, STEPS, TRIALS);
    
    int bs=256, ag=(NA+bs-1)/bs, rg=(NR+bs-1)/bs;
    A *da; R *dr; cudaMalloc(&da, NA*sizeof(A)); cudaMalloc(&dr, NR*sizeof(R));
    srand(time(NULL));
    
    // Control vs multi-objective
    printf("Mode                      Total     PerAgent  Explored\n");
    printf("-----------------------------------------------------\n");
    
    int ctrl_tot=0;
    for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000;
    reset<<<1,1>>>();initA<<<ag,bs>>>(da,NA,s);initR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
    for(int step=0;step<STEPS;step++){step_collect<<<ag,bs>>>(da,NA,dr,NR,s,step);respawn<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
    int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));ctrl_tot+=h;}
    printf("%-26s%-10d%-10.2f\n","Collect only (control)",ctrl_tot/TRIALS,(float)(ctrl_tot/TRIALS)/NA);
    
    // Fixed allocation sweep
    struct { float c, e; } allocs[] = {{1.0f,0.0f},{0.9f,0.05f},{0.8f,0.1f},{0.7f,0.2f},{0.6f,0.3f},{0.5f,0.4f}};
    for(int ai=0;ai<6;ai++){
        int tot=0,exp=0;
        for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+ai*10000;
        reset<<<1,1>>>();initA<<<ag,bs>>>(da,NA,s);initR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
        for(int step=0;step<STEPS;step++){step_multi_fixed<<<ag,bs>>>(da,NA,dr,NR,s,step,allocs[ai].c,allocs[ai].e,0.3f);respawn<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
        int h1,h2;cudaMemcpyFromSymbol(&h1,d_tot,sizeof(int));cudaMemcpyFromSymbol(&h2,d_explored,sizeof(int));tot+=h1;exp+=h2;}
        printf("Fixed C=%.1f E=%.1f R=%.1f       %-10d%-10.2f%-10d\n",allocs[ai].c,allocs[ai].e,1.0f-allocs[ai].c-allocs[ai].e,tot/TRIALS,(float)(tot/TRIALS)/NA,exp/TRIALS);
    }
    
    // Adaptive
    int adapt_tot=0,adapt_exp=0;
    for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+99999;
    reset<<<1,1>>>();initA<<<ag,bs>>>(da,NA,s);initR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
    for(int step=0;step<STEPS;step++){step_multi_adaptive<<<ag,bs>>>(da,NA,dr,NR,s,step,0.3f);respawn<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
    int h1,h2;cudaMemcpyFromSymbol(&h1,d_tot,sizeof(int));cudaMemcpyFromSymbol(&h2,d_explored,sizeof(int));adapt_tot+=h1;adapt_exp+=h2;}
    printf("%-26s%-10d%-10.2f%-10d\n","Adaptive (energy-based)",adapt_tot/TRIALS,(float)(adapt_tot/TRIALS)/NA,adapt_exp/TRIALS);
    
    // Energy cost sweep (adaptive mode)
    printf("\n=== Energy Cost Sweep (adaptive mode) ===\n");
    printf("Cost    Total     PerAgent\n");
    printf("---------------------------\n");
    float costs[]={0.0f,0.1f,0.2f,0.3f,0.5f,1.0f,2.0f};
    for(int ci=0;ci<7;ci++){
        int tot=0;
        for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+ci*8000;
        reset<<<1,1>>>();initA<<<ag,bs>>>(da,NA,s);initR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
        for(int step=0;step<STEPS;step++){step_multi_adaptive<<<ag,bs>>>(da,NA,dr,NR,s,step,costs[ci]);respawn<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
        int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));tot+=h;}
        printf("%-7.1f%-10d%-10.2f\n",costs[ci],tot/TRIALS,(float)(tot/TRIALS)/NA);
    }
    
    cudaFree(da);cudaFree(dr);return 0;
}
