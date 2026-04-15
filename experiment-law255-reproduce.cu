// experiment-law255-reproduce.cu
// EXACT reproduction of Law 255 parameters: 128 agents, 64 grid, 400 food, 1500 steps
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NF 128
#define FOOD 400
#define W 256.0f
#define GRID 64
#define BLK 128
#define STEPS 1500
#define TRIALS 10

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void phase1_traces(int *traces, int steps, int n, float w, int grid_w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng) * w, y = cr(&rng) * w;
    for (int t = 0; t < steps; t++) {
        float dx = (cr(&rng) - 0.5f) * 6.0f;
        float dy = (cr(&rng) - 0.5f) * 6.0f;
        x = fmodf(x + dx + w, w);
        y = fmodf(y + dy + w, w);
        int gx = (int)(x / w * grid_w) % grid_w;
        int gy = (int)(y / w * grid_w) % grid_w;
        atomicAdd(&traces[gy * grid_w + gx], 1);
    }
}

// mode 0: ignore (random), mode 1: toward traces, mode 2: away from traces, mode 3: oracle
__global__ void phase2_exploit(float *scores, int *traces, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, float w, int grid_w, int mode, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + (tid + n) * 997;
    float x = cr(&rng) * w, y = cr(&rng) * w;
    float energy = 150.0f, score = 0.0f;
    float base_angle = (tid + n) * 2.39996f;
    float dir[8];
    for (int i = 0; i < 8; i++) dir[i] = base_angle + i * 0.785f;
    
    for (int t = 0; t < steps && energy > 0; t++) {
        int p = t % 8;
        float dx = cosf(dir[p]) * 2.0f, dy = sinf(dir[p]) * 2.0f;
        
        if (mode == 1 || mode == 2) {
            int gx = (int)(x / w * grid_w) % grid_w;
            int gy = (int)(y / w * grid_w) % grid_w;
            float best_v = 0, best_dx = 0, best_dy = 0;
            for (int dy2 = -2; dy2 <= 2; dy2++) {
                for (int dx2 = -2; dx2 <= 2; dx2++) {
                    if (dx2 == 0 && dy2 == 0) continue;
                    int nx = (gx + dx2 + grid_w) % grid_w;
                    int ny = (gy + dy2 + grid_w) % grid_w;
                    int v = traces[ny * grid_w + nx];
                    if (mode == 2) v = -v;
                    if (v > best_v) {
                        best_v = v;
                        best_dx = dx2 * w / grid_w;
                        best_dy = dy2 * w / grid_w;
                    }
                }
            }
            dx += best_dx * 0.1f;
            dy += best_dy * 0.1f;
        } else if (mode == 3) {
            float best_d = 999999.0f;
            for (int i = 0; i < food_count; i += 8) {
                if (!falive[i]) continue;
                float fdx = fx[i] - x, fdy = fy[i] - y;
                if (fdx > w/2) fdx -= w; if (fdx < -w/2) fdx += w;
                if (fdy > w/2) fdy -= w; if (fdy < -w/2) fdy += w;
                float d = fdx*fdx + fdy*fdy;
                if (d < best_d) { best_d = d; dx = (fx[i]-x)*0.3f; dy = (fy[i]-y)*0.3f; }
            }
        }
        
        dx += (cr(&rng) - 0.5f) * 2.0f;
        dy += (cr(&rng) - 0.5f) * 2.0f;
        float dist = sqrtf(dx*dx + dy*dy);
        energy -= 0.005f + dist * 0.003f;
        x = fmodf(x + dx + w, w);
        y = fmodf(y + dy + w, w);
        
        for (int i = 0; i < food_count; i++) {
            if (!falive[i]) continue;
            float fdx = fx[i] - x, fdy = fy[i] - y;
            if (fdx > w/2) fdx -= w; if (fdx < -w/2) fdx += w;
            if (fdy > w/2) fdy -= w; if (fdy < -w/2) fdy += w;
            if (fdx*fdx + fdy*fdy < 16.0f) {
                int old = atomicExch(&falive[i], 0);
                if (old) { energy = fminf(energy + 10.0f, 200.0f); score += 1.0f; }
            }
        }
    }
    scores[tid] = score;
}

int main() {
    printf("=== Law 255 REPRODUCTION: Exact Original Parameters ===\n");
    printf("Phase1=%d agents, Grid=%dx%d, Food=%d, Steps=%d, Trials=%d\n\n", NF, GRID, GRID, FOOD, STEPS, TRIALS);
    
    const char* modes[] = {"Random(Ignore)", "Toward", "Away(coverage)", "Oracle"};
    float totals[4] = {0, 0, 0, 0};
    
    int blk = (NF + BLK - 1) / BLK;
    int *d_tr, *d_fa;
    float *d_s, *d_fx, *d_fy;
    cudaMalloc(&d_tr, GRID * GRID * sizeof(int));
    cudaMalloc(&d_s, NF * sizeof(float));
    cudaMalloc(&d_fx, FOOD * sizeof(float));
    cudaMalloc(&d_fy, FOOD * sizeof(float));
    cudaMalloc(&d_fa, FOOD * sizeof(int));
    
    for (int trial = 0; trial < TRIALS; trial++) {
        float hfx[FOOD], hfy[FOOD];
        srand(42 + trial * 777);
        for (int i = 0; i < FOOD; i++) {
            hfx[i] = ((float)rand() / RAND_MAX) * W;
            hfy[i] = ((float)rand() / RAND_MAX) * W;
        }
        cudaMemcpy(d_fx, hfx, FOOD * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy, hfy, FOOD * sizeof(float), cudaMemcpyHostToDevice);
        
        for (int mode = 0; mode < 4; mode++) {
            cudaMemset(d_tr, 0, GRID * GRID * sizeof(int));
            int ones[FOOD]; for (int i = 0; i < FOOD; i++) ones[i] = 1;
            cudaMemcpy(d_fa, ones, FOOD * sizeof(int), cudaMemcpyHostToDevice);
            
            if (mode == 1 || mode == 2) {
                phase1_traces<<<blk, BLK>>>(d_tr, STEPS, NF, W, GRID, (unsigned int)(42 + trial * 1111));
            }
            
            phase2_exploit<<<blk, BLK>>>(d_s, d_tr, d_fx, d_fy, d_fa, STEPS, NF, FOOD, W, GRID, mode,
                (unsigned int)(42 + trial * 1111));
            cudaDeviceSynchronize();
            
            float h_s[NF];
            cudaMemcpy(h_s, d_s, NF * sizeof(float), cudaMemcpyDeviceToHost);
            float total = 0;
            for (int i = 0; i < NF; i++) total += h_s[i];
            totals[mode] += total / NF;
        }
    }
    
    for (int m = 0; m < 4; m++) totals[m] /= TRIALS;
    
    printf("Results (avg food/agent):\n");
    for (int m = 0; m < 4; m++) printf("  %s: %.3f\n", modes[m], totals[m]);
    
    float away_boost = (totals[2] - totals[0]) / totals[0];
    float toward_hurt = (totals[1] - totals[0]) / totals[0];
    float oracle_gap = (totals[3] - totals[2]) / (totals[3] - totals[0]);
    
    printf("\nAway boost over random: %.1f%%\n", away_boost * 100);
    printf("Toward vs random: %.1f%%\n", toward_hurt * 100);
    printf("Oracle gap (const): %.4f\n", oracle_gap);
    
    // Also check trace coverage
    int h_tr[GRID * GRID];
    int *d_tr2; cudaMalloc(&d_tr2, GRID*GRID*sizeof(int));
    cudaMemset(d_tr2, 0, GRID*GRID*sizeof(int));
    phase1_traces<<<blk, BLK>>>(d_tr2, STEPS, NF, W, GRID, 42);
    cudaMemcpy(h_tr, d_tr2, GRID*GRID*sizeof(int), cudaMemcpyDeviceToHost);
    int visited = 0, max_trace = 0;
    for (int i = 0; i < GRID*GRID; i++) { if (h_tr[i] > 0) visited++; if (h_tr[i] > max_trace) max_trace = h_tr[i]; }
    printf("\nTrace coverage: %d/%d cells (%.1f%%), max trace value: %d\n", visited, GRID*GRID, 100.0*visited/(GRID*GRID), max_trace);
    
    return 0;
}
