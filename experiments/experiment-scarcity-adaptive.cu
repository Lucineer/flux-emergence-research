// experiment-scarcity-adaptive.cu
// Scarcity-Adaptive Protocol — agents that adjust behavior based on
// local resource density, not global instructions.
//
// Bering Sea connection: the ensign doesn't get told "watch engineering."
// He watches engineering because THAT'S WHERE THE RATE OF CHANGE IS.
//
// Hypothesis: agents that adapt their grab range, speed, and cooperation
// based on local scarcity outperform agents with fixed parameters.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID        256
#define NUM_AGENTS  2048
#define NUM_FOOD    300
#define STEPS       3000
#define TRIALS      5

struct Agent {
    float x, y;
    float grab_range;
    float speed;
    float base_grab;    // genetic grab range (fixed)
    float collected;
    int local_density;  // food count in perception radius
    float adaptation;   // how much they've adapted (0.0 to 1.0)
};

struct Food {
    float x, y;
    int active;
};

__device__ int d_total;
__device__ int d_adapted_agents;  // agents that changed behavior

__device__ float lcg(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}

__global__ void reset() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_total = 0;
        d_adapted_agents = 0;
    }
}

__global__ void init_agents(Agent *a, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int s = seed + i * 7919;
    a[i].x = lcg(&s) * GRID;
    a[i].y = lcg(&s) * GRID;
    a[i].base_grab = 2.0f + lcg(&s) * 2.0f;  // 2.0-4.0
    a[i].grab_range = a[i].base_grab;
    a[i].speed = 2.0f + lcg(&s) * 2.0f;
    a[i].collected = 0;
    a[i].local_density = 0;
    a[i].adaptation = 0.0f;
}

__global__ void init_food(Food *f, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int s = seed + i * 65537;
    f[i].x = lcg(&s) * GRID;
    f[i].y = lcg(&s) * GRID;
    f[i].active = 1;
}

// MODE 0: Fixed parameters (control)
__global__ void step_fixed(Agent *a, int na, Food *f, int nf, unsigned int seed, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;
    unsigned int s = seed + i * 131 + step * 997;
    
    a[i].x = fmodf(a[i].x + (lcg(&s)-0.5f) * a[i].speed * 2 + GRID, GRID);
    a[i].y = fmodf(a[i].y + (lcg(&s)-0.5f) * a[i].speed * 2 + GRID, GRID);
    
    float g2 = a[i].grab_range * a[i].grab_range;
    for (int j = 0; j < nf; j++) {
        if (!f[j].active) continue;
        float dx = a[i].x - f[j].x, dy = a[i].y - f[j].y;
        if (dx > GRID*0.5f) dx -= GRID; if (dx < -GRID*0.5f) dx += GRID;
        if (dy > GRID*0.5f) dy -= GRID; if (dy < -GRID*0.5f) dy += GRID;
        if (dx*dx + dy*dy < g2) {
            f[j].active = 0;
            a[i].collected++;
            atomicAdd(&d_total, 1);
        }
    }
}

// MODE 1: Scarcity-adaptive (adjust grab range based on local density)
__global__ void step_adaptive(Agent *a, int na, Food *f, int nf, unsigned int seed, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;
    unsigned int s = seed + i * 131 + step * 997;
    
    // Count local food density (perception radius = 20)
    float perception = 20.0f;
    int local_food = 0;
    for (int j = 0; j < nf; j++) {
        if (!f[j].active) continue;
        float dx = a[i].x - f[j].x, dy = a[i].y - f[j].y;
        if (dx > GRID*0.5f) dx -= GRID; if (dx < -GRID*0.5f) dx += GRID;
        if (dy > GRID*0.5f) dy -= GRID; if (dy < -GRID*0.5f) dy += GRID;
        if (dx*dx + dy*dy < perception*perception) local_food++;
    }
    a[i].local_density = local_food;
    
    // Adapt: scarce food → expand grab range, abundant → contract (save energy)
    float target_grab;
    if (local_food == 0) {
        target_grab = a[i].base_grab * 2.0f;  // double range when starving
    } else if (local_food < 3) {
        target_grab = a[i].base_grab * 1.5f;
    } else if (local_food > 10) {
        target_grab = a[i].base_grab * 0.7f;  // contract when abundant
    } else {
        target_grab = a[i].base_grab;
    }
    
    // Smooth adaptation
    float adapt_rate = 0.1f;
    a[i].grab_range += (target_grab - a[i].grab_range) * adapt_rate;
    a[i].grab_range = fmaxf(1.0f, fminf(a[i].grab_range, 8.0f));
    
    float adapt_delta = fabsf(a[i].grab_range - a[i].base_grab);
    if (adapt_delta > 0.5f) atomicAdd(&d_adapted_agents, 1);
    
    // Move toward nearest food if scarce, random if abundant
    float dx = 0, dy = 0;
    if (local_food < 5) {
        // Move toward food
        float best_d2 = perception * perception;
        for (int j = 0; j < nf; j++) {
            if (!f[j].active) continue;
            float fdx = f[j].x - a[i].x, fdy = f[j].y - a[i].y;
            if (fdx > GRID*0.5f) fdx -= GRID; if (fdx < -GRID*0.5f) fdx += GRID;
            if (fdy > GRID*0.5f) fdy -= GRID; if (fdy < -GRID*0.5f) fdy += GRID;
            float d2 = fdx*fdx + fdy*fdy;
            if (d2 < best_d2) { best_d2 = d2; dx = fdx; dy = fdy; }
        }
        if (best_d2 > 0) {
            float d = sqrtf(best_d2);
            dx = dx / d * a[i].speed;
            dy = dy / d * a[i].speed;
        } else {
            dx = (lcg(&s)-0.5f) * a[i].speed * 2;
            dy = (lcg(&s)-0.5f) * a[i].speed * 2;
        }
    } else {
        dx = (lcg(&s)-0.5f) * a[i].speed * 2;
        dy = (lcg(&s)-0.5f) * a[i].speed * 2;
    }
    
    a[i].x = fmodf(a[i].x + dx + GRID, GRID);
    a[i].y = fmodf(a[i].y + dy + GRID, GRID);
    
    float g2 = a[i].grab_range * a[i].grab_range;
    for (int j = 0; j < nf; j++) {
        if (!f[j].active) continue;
        float fdx = a[i].x - f[j].x, fdy = a[i].y - f[j].y;
        if (fdx > GRID*0.5f) fdx -= GRID; if (fdx < -GRID*0.5f) fdx += GRID;
        if (fdy > GRID*0.5f) fdy -= GRID; if (fdy < -GRID*0.5f) fdy += GRID;
        if (fdx*fdx + fdy*fdy < g2) {
            f[j].active = 0;
            a[i].collected++;
            atomicAdd(&d_total, 1);
        }
    }
}

__global__ void respawn(Food *f, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || f[i].active) return;
    unsigned int s = seed + i * 3571;
    f[i].active = 1;
    f[i].x = lcg(&s) * GRID;
    f[i].y = lcg(&s) * GRID;
}

int main() {
    printf("=== Scarcity-Adaptive Agents ===\n");
    printf("Agents: %d | Food: %d | Grid: %d | Steps: %d\n\n", NUM_AGENTS, NUM_FOOD, GRID, STEPS);
    
    int bs = 256, ag = (NUM_AGENTS+bs-1)/bs, fg = (NUM_FOOD+bs-1)/bs;
    Agent *da; Food *df;
    cudaMalloc(&da, NUM_AGENTS * sizeof(Agent));
    cudaMalloc(&df, NUM_FOOD * sizeof(Food));
    srand(time(NULL));
    
    // Sweep food scarcity
    printf("=== Food Scarcity Sweep ===\n");
    printf("%-8s %-12s %-12s %-12s %-12s\n", "Food", "Fixed", "Adaptive", "Lift", "Adapted%");
    printf("%s\n", "----------------------------------------------------");
    
    int foods[] = {50, 100, 200, 400, 800};
    for (int fi = 0; fi < 5; fi++) {
        int nf = foods[fi];
        int nfg = (nf+bs-1)/bs;
        Food *df2; cudaMalloc(&df2, nf * sizeof(Food));
        
        int fixed_tot = 0, adapt_tot = 0, adapted = 0;
        
        for (int t = 0; t < TRIALS; t++) {
            unsigned int seed = (unsigned int)time(NULL) + t*50000 + fi*100000;
            
            // Fixed
            reset<<<1,1>>>(); init_agents<<<ag,bs>>>(da,NUM_AGENTS,seed);
            init_food<<<nfg,bs>>>(df2,nf,seed+1); cudaDeviceSynchronize();
            for (int s=0; s<STEPS; s++) {
                step_fixed<<<ag,bs>>>(da,NUM_AGENTS,df2,nf,seed,s);
                respawn<<<nfg,bs>>>(df2,nf,seed+s); cudaDeviceSynchronize();
            }
            int h; cudaMemcpyFromSymbol(&h,d_total,sizeof(int)); fixed_tot += h;
            
            // Adaptive
            reset<<<1,1>>>(); init_agents<<<ag,bs>>>(da,NUM_AGENTS,seed+999);
            init_food<<<nfg,bs>>>(df2,nf,seed+1000); cudaDeviceSynchronize();
            for (int s=0; s<STEPS; s++) {
                step_adaptive<<<ag,bs>>>(da,NUM_AGENTS,df2,nf,seed+999,s);
                respawn<<<nfg,bs>>>(df2,nf,seed+999+s); cudaDeviceSynchronize();
            }
            cudaMemcpyFromSymbol(&h,d_total,sizeof(int)); adapt_tot += h;
            int ha; cudaMemcpyFromSymbol(&ha,d_adapted_agents,sizeof(int)); adapted += ha;
        }
        
        float lift = (float)adapt_tot / fixed_tot;
        float adapt_pct = (float)(adapted/TRIALS) / (NUM_AGENTS * STEPS) * 100;
        printf("%-8d %-12d %-12d %-12.2fx %-12.1f%%\n", nf, fixed_tot/TRIALS, adapt_tot/TRIALS, lift, adapt_pct);
        cudaFree(df2);
    }
    
    // Sweep agent density
    printf("\n=== Agent Density Sweep (food=200) ===\n");
    printf("%-8s %-12s %-12s %-12s\n", "Agents", "Fixed", "Adaptive", "Lift");
    printf("%s\n", "----------------------------------------");
    
    int agents[] = {256, 512, 1024, 2048, 4096};
    for (int ai = 0; ai < 5; ai++) {
        int na = agents[ai];
        int nag = (na+bs-1)/bs;
        Agent *da2; cudaMalloc(&da2, na * sizeof(Agent));
        
        int fixed_tot = 0, adapt_tot = 0;
        for (int t = 0; t < TRIALS; t++) {
            unsigned int seed = (unsigned int)time(NULL) + t*50000 + ai*100000;
            reset<<<1,1>>>(); init_agents<<<nag,bs>>>(da2,na,seed);
            init_food<<<fg,bs>>>(df,NUM_FOOD,seed+1); cudaDeviceSynchronize();
            for (int s=0; s<STEPS; s++) {
                step_fixed<<<nag,bs>>>(da2,na,df,NUM_FOOD,seed,s);
                respawn<<<fg,bs>>>(df,NUM_FOOD,seed+s); cudaDeviceSynchronize();
            }
            int h; cudaMemcpyFromSymbol(&h,d_total,sizeof(int)); fixed_tot += h;
            
            reset<<<1,1>>>(); init_agents<<<nag,bs>>>(da2,na,seed+999);
            init_food<<<fg,bs>>>(df,NUM_FOOD,seed+1000); cudaDeviceSynchronize();
            for (int s=0; s<STEPS; s++) {
                step_adaptive<<<nag,bs>>>(da2,na,df,NUM_FOOD,seed+999,s);
                respawn<<<fg,bs>>>(df,NUM_FOOD,seed+999+s); cudaDeviceSynchronize();
            }
            cudaMemcpyFromSymbol(&h,d_total,sizeof(int)); adapt_tot += h;
        }
        printf("%-8d %-12d %-12d %-12.2fx\n", na, fixed_tot/TRIALS, adapt_tot/TRIALS, (float)adapt_tot/fixed_tot);
        cudaFree(da2);
    }
    
    cudaFree(da); cudaFree(df);
    return 0;
}
