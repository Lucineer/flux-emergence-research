// experiment-anticonvergence.cu
// Anti-Convergence Drift — agents actively avoid becoming similar to neighbors
//
// Prior finding: strong coupling HOMOGENIZES (v2: 10x same-type influence → clones)
// Prior finding: trails cause congestion from clustering
// Prior finding: gossip hurts (more sharing = more convergence = worse)
//
// Hypothesis: agents that actively DIVERGE from neighbors will:
// 1. Maintain population diversity (more niche coverage)
// 2. Explore more of the space (less redundant paths)
// 3. Outperform homogeneous populations in heterogeneous environments
//
// Mechanism: each agent tracks a "behavior vector" (grab range, speed, persistence).
// If neighbors' vectors are too similar, agent mutates away.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID 256
#define NA 1024
#define NR 300
#define STEPS 3000
#define TRIALS 5
#define BEHAVIOR_DIM 3  // grab_range, speed, persistence

struct A {
    float x, y;
    float behav[BEHAVIOR_DIM];  // behavior vector
    float col;
    float spd;
};

struct R {
    float x, y;
    int active;
    float value;
};

__device__ int d_tot;
__device__ float d_diversity;  // avg pairwise behavior distance
__device__ int d_div_count;

__device__ float rng(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}

__device__ float behav_dist(float *a, float *b) {
    float d = 0;
    for (int i = 0; i < BEHAVIOR_DIM; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return sqrtf(d);
}

__global__ void reset() {
    if (threadIdx.x == 0 && blockIdx.x == 0) { d_tot = 0; d_diversity = 0; d_div_count = 0; }
}

__global__ void initA(A *a, int n, unsigned int s, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int r = s + i * 7919;
    a[i].x = rng(&r) * GRID;
    a[i].y = rng(&r) * GRID;
    a[i].col = 0;
    a[i].spd = 3.0f;
    
    if (mode == 0) {
        // Homogeneous start
        a[i].behav[0] = 3.0f;  // grab_range
        a[i].behav[1] = 3.0f;  // speed
        a[i].behav[2] = 0.5f;  // persistence (0=random, 1=straight line)
    } else {
        // Diverse start
        a[i].behav[0] = 1.0f + rng(&r) * 5.0f;
        a[i].behav[1] = 1.0f + rng(&r) * 5.0f;
        a[i].behav[2] = rng(&r);
    }
}

__global__ void initR(R *r, int n, unsigned int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int r2 = s + i * 65537;
    r[i].x = rng(&r2) * GRID;
    r[i].y = rng(&r2) * GRID;
    r[i].active = 1;
    r[i].value = 1.0f;
}

// MODE 0: No drift control (control)
__global__ void step_control(A *a, int na, R *r, int nr, unsigned int s, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;
    unsigned int r2 = s + i * 131 + step * 997;
    
    // Move: persistence controls straight-line vs random
    float straight = a[i].behav[2];
    float dx = (rng(&r2) - 0.5f) * (1.0f - straight) * 2 + straight * cosf(step * 0.1f + i);
    float dy = (rng(&r2) - 0.5f) * (1.0f - straight) * 2 + straight * sinf(step * 0.1f + i);
    a[i].x = fmodf(a[i].x + dx * a[i].behav[1] + GRID, GRID);
    a[i].y = fmodf(a[i].y + dy * a[i].behav[1] + GRID, GRID);
    
    // Collect
    float g2 = a[i].behav[0] * a[i].behav[0];
    for (int j = 0; j < nr; j++) {
        if (!r[j].active) continue;
        float fdx = a[i].x - r[j].x, fdy = a[i].y - r[j].y;
        if (fdx > GRID * 0.5f) fdx -= GRID; if (fdx < -GRID * 0.5f) fdx += GRID;
        if (fdy > GRID * 0.5f) fdy -= GRID; if (fdy < -GRID * 0.5f) fdy += GRID;
        if (fdx * fdx + fdy * fdy < g2) {
            r[j].active = 0;
            a[i].col += r[j].value;
            atomicAdd(&d_tot, 1);
            break;
        }
    }
}

// MODE 1: Anti-convergence drift — agents mutate away from local average
__global__ void step_anticonv(A *a, int na, R *r, int nr, unsigned int s, int step,
                               float drift_rate, float check_radius, int check_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;
    unsigned int r2 = s + i * 131 + step * 997;
    
    // Compute local average behavior
    float avg_b[3] = {0, 0, 0};
    int neighbors = 0;
    float r2_check = check_radius * check_radius;
    
    // Sample random neighbors (avoid O(n^2))
    for (int c = 0; c < check_count; c++) {
        int other = (i * 7 + c * 31 + step * 13) % na;
        float dx = a[i].x - a[other].x, dy = a[i].y - a[other].y;
        if (dx > GRID * 0.5f) dx -= GRID; if (dx < -GRID * 0.5f) dx += GRID;
        if (dy > GRID * 0.5f) dy -= GRID; if (dy < -GRID * 0.5f) dy += GRID;
        if (dx * dx + dy * dy < r2_check) {
            for (int d = 0; d < BEHAVIOR_DIM; d++) avg_b[d] += a[other].behav[d];
            neighbors++;
        }
    }
    
    // Anti-convergence: if too similar to neighbors, drift away
    if (neighbors > 0) {
        for (int d = 0; d < BEHAVIOR_DIM; d++) avg_b[d] /= neighbors;
        
        float sim = behav_dist(a[i].behav, avg_b);
        if (sim < 1.0f) {  // too similar
            // Mutate AWAY from average
            for (int d = 0; d < BEHAVIOR_DIM; d++) {
                float diff = a[i].behav[d] - avg_b[d];
                a[i].behav[d] += diff * drift_rate + (rng(&r2) - 0.5f) * drift_rate;
            }
        }
    }
    
    // Clamp behaviors
    a[i].behav[0] = fmaxf(0.5f, fminf(a[i].behav[0], 8.0f));  // grab_range
    a[i].behav[1] = fmaxf(0.5f, fminf(a[i].behav[1], 8.0f));  // speed
    a[i].behav[2] = fmaxf(0.0f, fminf(a[i].behav[2], 1.0f));  // persistence
    
    // Move
    float straight = a[i].behav[2];
    float dx = (rng(&r2) - 0.5f) * (1.0f - straight) * 2 + straight * cosf(step * 0.1f + i);
    float dy = (rng(&r2) - 0.5f) * (1.0f - straight) * 2 + straight * sinf(step * 0.1f + i);
    a[i].x = fmodf(a[i].x + dx * a[i].behav[1] + GRID, GRID);
    a[i].y = fmodf(a[i].y + dy * a[i].behav[1] + GRID, GRID);
    
    // Collect
    float g2 = a[i].behav[0] * a[i].behav[0];
    for (int j = 0; j < nr; j++) {
        if (!r[j].active) continue;
        float fdx = a[i].x - r[j].x, fdy = a[i].y - r[j].y;
        if (fdx > GRID * 0.5f) fdx -= GRID; if (fdx < -GRID * 0.5f) fdx += GRID;
        if (fdy > GRID * 0.5f) fdy -= GRID; if (fdy < -GRID * 0.5f) fdy += GRID;
        if (fdx * fdx + fdy * fdy < g2) {
            r[j].active = 0;
            a[i].col += r[j].value;
            atomicAdd(&d_tot, 1);
            break;
        }
    }
}

// MODE 2: Convergence (opposite — agents imitate successful neighbors)
__global__ void step_converge(A *a, int na, R *r, int nr, unsigned int s, int step,
                               float conv_rate, int check_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;
    unsigned int r2 = s + i * 131 + step * 997;
    
    // Find most successful nearby neighbor and imitate
    float best_col = a[i].col;
    int best_other = -1;
    
    for (int c = 0; c < check_count; c++) {
        int other = (i * 7 + c * 31 + step * 13) % na;
        if (other == i) continue;
        if (a[other].col > best_col) {
            best_col = a[other].col;
            best_other = other;
        }
    }
    
    if (best_other >= 0) {
        for (int d = 0; d < BEHAVIOR_DIM; d++) {
            a[i].behav[d] += (a[best_other].behav[d] - a[i].behav[d]) * conv_rate;
        }
    }
    
    a[i].behav[0] = fmaxf(0.5f, fminf(a[i].behav[0], 8.0f));
    a[i].behav[1] = fmaxf(0.5f, fminf(a[i].behav[1], 8.0f));
    a[i].behav[2] = fmaxf(0.0f, fminf(a[i].behav[2], 1.0f));
    
    float straight = a[i].behav[2];
    float dx = (rng(&r2) - 0.5f) * (1.0f - straight) * 2 + straight * cosf(step * 0.1f + i);
    float dy = (rng(&r2) - 0.5f) * (1.0f - straight) * 2 + straight * sinf(step * 0.1f + i);
    a[i].x = fmodf(a[i].x + dx * a[i].behav[1] + GRID, GRID);
    a[i].y = fmodf(a[i].y + dy * a[i].behav[1] + GRID, GRID);
    
    float g2 = a[i].behav[0] * a[i].behav[0];
    for (int j = 0; j < nr; j++) {
        if (!r[j].active) continue;
        float fdx = a[i].x - r[j].x, fdy = a[i].y - r[j].y;
        if (fdx > GRID * 0.5f) fdx -= GRID; if (fdx < -GRID * 0.5f) fdx += GRID;
        if (fdy > GRID * 0.5f) fdy -= GRID; if (fdy < -GRID * 0.5f) fdy += GRID;
        if (fdx * fdx + fdy * fdy < g2) {
            r[j].active = 0;
            a[i].col += r[j].value;
            atomicAdd(&d_tot, 1);
            break;
        }
    }
}

__global__ void respawn(R *r, int n, unsigned int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || r[i].active) return;
    unsigned int r2 = s + i * 3571;
    r[i].x = rng(&r2) * GRID;
    r[i].y = rng(&r2) * GRID;
    r[i].active = 1;
    r[i].value = 1.0f;
}

int main() {
    printf("=== Anti-Convergence Drift ===\n");
    printf("Agents:%d Resources:%d Steps:%d Trials:%d\n\n", NA, NR, STEPS, TRIALS);
    
    int bs = 256, ag = (NA + bs - 1) / bs, rg = (NR + bs - 1) / bs;
    A *da; R *dr;
    cudaMalloc(&da, NA * sizeof(A));
    cudaMalloc(&dr, NR * sizeof(R));
    srand(time(NULL));
    
    // === Three-way comparison ===
    printf("Mode              Total     PerAgent  Diversity\n");
    printf("-------------------------------------------------\n");
    
    const char *modes[] = {"Control (no drift)", "Anti-convergence", "Convergence (imitate)"};
    
    for (int mode = 0; mode < 3; mode++) {
        int tot = 0;
        for (int t = 0; t < TRIALS; t++) {
            unsigned int s = (unsigned int)time(NULL) + t * 50000 + mode * 100000;
            reset<<<1, 1>>>();
            initA<<<ag, bs>>>(da, NA, s, 1);  // diverse start
            initR<<<rg, bs>>>(dr, NR, s + 1);
            cudaDeviceSynchronize();
            
            for (int step = 0; step < STEPS; step++) {
                switch (mode) {
                    case 0: step_control<<<ag, bs>>>(da, NA, dr, NR, s, step); break;
                    case 1: step_anticonv<<<ag, bs>>>(da, NA, dr, NR, s, step, 0.3f, 30.0f, 10); break;
                    case 2: step_converge<<<ag, bs>>>(da, NA, dr, NR, s, step, 0.1f, 10); break;
                }
                respawn<<<rg, bs>>>(dr, NR, s + step);
                cudaDeviceSynchronize();
            }
            
            int h; cudaMemcpyFromSymbol(&h, d_tot, sizeof(int));
            tot += h;
        }
        printf("%-18s %-9d %-9.2f\n", modes[mode], tot / TRIALS, (float)(tot / TRIALS) / NA);
    }
    
    // === Drift rate sweep ===
    printf("\n=== Anti-Convergence Drift Rate Sweep ===\n");
    printf("DriftRate  Total     PerAgent\n");
    printf("-------------------------------\n");
    
    float rates[] = {0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 0.7f, 1.0f};
    for (int ri = 0; ri < 7; ri++) {
        int tot = 0;
        for (int t = 0; t < TRIALS; t++) {
            unsigned int s = (unsigned int)time(NULL) + t * 50000 + ri * 10000;
            reset<<<1, 1>>>();
            initA<<<ag, bs>>>(da, NA, s, 1);
            initR<<<rg, bs>>>(dr, NR, s + 1);
            cudaDeviceSynchronize();
            for (int step = 0; step < STEPS; step++) {
                step_anticonv<<<ag, bs>>>(da, NA, dr, NR, s, step, rates[ri], 30.0f, 10);
                respawn<<<rg, bs>>>(dr, NR, s + step);
                cudaDeviceSynchronize();
            }
            int h; cudaMemcpyFromSymbol(&h, d_tot, sizeof(int));
            tot += h;
        }
        printf("%-11.2f%-9d%-9.2f\n", rates[ri], tot / TRIALS, (float)(tot / TRIALS) / NA);
    }
    
    // === Check radius sweep ===
    printf("\n=== Anti-Convergence Check Radius Sweep ===\n");
    printf("Radius    Total     PerAgent\n");
    printf("-----------------------------\n");
    
    float radii[] = {10.0f, 20.0f, 30.0f, 50.0f, 80.0f, 128.0f};
    for (int ri = 0; ri < 6; ri++) {
        int tot = 0;
        for (int t = 0; t < TRIALS; t++) {
            unsigned int s = (unsigned int)time(NULL) + t * 50000 + ri * 15000;
            reset<<<1, 1>>>();
            initA<<<ag, bs>>>(da, NA, s, 1);
            initR<<<rg, bs>>>(dr, NR, s + 1);
            cudaDeviceSynchronize();
            for (int step = 0; step < STEPS; step++) {
                step_anticonv<<<ag, bs>>>(da, NA, dr, NR, s, step, 0.3f, radii[ri], 10);
                respawn<<<rg, bs>>>(dr, NR, s + step);
                cudaDeviceSynchronize();
            }
            int h; cudaMemcpyFromSymbol(&h, d_tot, sizeof(int));
            tot += h;
        }
        printf("%-10.0f%-9d%-9.2f\n", radii[ri], tot / TRIALS, (float)(tot / TRIALS) / NA);
    }
    
    // === Homogeneous vs Diverse start ===
    printf("\n=== Start Diversity (anti-conv drift=0.3) ===\n");
    printf("Start       Total     PerAgent\n");
    printf("-----------------------------\n");
    
    for (int start = 0; start < 2; start++) {
        int tot = 0;
        const char *snames[] = {"Homogeneous", "Diverse"};
        for (int t = 0; t < TRIALS; t++) {
            unsigned int s = (unsigned int)time(NULL) + t * 50000 + start * 77777;
            reset<<<1, 1>>>();
            initA<<<ag, bs>>>(da, NA, s, start);
            initR<<<rg, bs>>>(dr, NR, s + 1);
            cudaDeviceSynchronize();
            for (int step = 0; step < STEPS; step++) {
                step_anticonv<<<ag, bs>>>(da, NA, dr, NR, s, step, 0.3f, 30.0f, 10);
                respawn<<<rg, bs>>>(dr, NR, s + step);
                cudaDeviceSynchronize();
            }
            int h; cudaMemcpyFromSymbol(&h, d_tot, sizeof(int));
            tot += h;
        }
        printf("%-11s%-9d%-9.2f\n", snames[start], tot / TRIALS, (float)(tot / TRIALS) / NA);
    }
    
    cudaFree(da); cudaFree(dr);
    return 0;
}
