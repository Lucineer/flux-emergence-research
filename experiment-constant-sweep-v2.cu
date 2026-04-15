// experiment-constant-sweep-v2.cu
// Law 267 CORRECTED: Two-phase design matching original Laws 255-265
// Phase 1: Build trace map (spatial history)
// Phase 2: Exploit (or ignore) trace map to find food
// Key question: Does the coverage constant hold across parameters?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_AGENTS 2048
#define MAX_FOOD 4096
#define MAX_GRID 256
#define BLK 128
#define PHASE_STEPS 1500
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

// Phase 1: Random walkers build trace map
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

// Phase 2: Exploit traces to find food
// mode 0: ignore traces (random)
// mode 1: steer toward high-trace areas (wrong — tests if traces hurt)
// mode 2: steer AWAY from high-trace areas (correct — coverage optimization)
// mode 3: steer toward nearest food (oracle)
__global__ void phase2_exploit(float *scores, int *traces, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, float w, int grid_w, int mode, float grab_r, unsigned int seed) {
    
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
            // Use trace map to steer
            int gx = (int)(x / w * grid_w) % grid_w;
            int gy = (int)(y / w * grid_w) % grid_w;
            float best_v = 0, best_dx = 0, best_dy = 0;
            for (int dy2 = -3; dy2 <= 3; dy2++) {
                for (int dx2 = -3; dx2 <= 3; dx2++) {
                    if (dx2 == 0 && dy2 == 0) continue;
                    int nx = (gx + dx2 + grid_w) % grid_w;
                    int ny = (gy + dy2 + grid_w) % grid_w;
                    int v = traces[ny * grid_w + nx];
                    if (mode == 2) v = -v; // AWAY from traces = toward unexplored
                    if (v > best_v) {
                        best_v = v;
                        best_dx = dx2 * w / grid_w;
                        best_dy = dy2 * w / grid_w;
                    }
                }
            }
            dx += best_dx * 0.15f;
            dy += best_dy * 0.15f;
        } else if (mode == 3) {
            // Oracle: steer toward nearest food
            float best_d = 999999.0f, bx = x, by = y;
            for (int i = 0; i < food_count; i += 4) {
                if (!falive[i]) continue;
                float fdx = fx[i] - x, fdy = fy[i] - y;
                if (fdx > w/2) fdx -= w; if (fdx < -w/2) fdx += w;
                if (fdy > w/2) fdy -= w; if (fdy < -w/2) fdy += w;
                float d = fdx*fdx + fdy*fdy;
                if (d < best_d) { best_d = d; bx = fx[i]; by = fy[i]; }
            }
            dx = (bx - x) * 0.3f;
            dy = (by - y) * 0.3f;
        }
        // mode 0: pure scripted walk (ignore traces)
        
        dx += (cr(&rng) - 0.5f) * 2.0f;
        dy += (cr(&rng) - 0.5f) * 2.0f;
        float dist = sqrtf(dx*dx + dy*dy);
        energy -= 0.005f + dist * 0.003f;
        x = fmodf(x + dx + w, w);
        y = fmodf(y + dy + w, w);
        
        // Collect food
        float gr2 = grab_r * grab_r;
        for (int i = 0; i < food_count; i++) {
            if (!falive[i]) continue;
            float fdx = fx[i] - x, fdy = fy[i] - y;
            if (fdx > w/2) fdx -= w; if (fdx < -w/2) fdx += w;
            if (fdy > w/2) fdy -= w; if (fdy < -w/2) fdy += w;
            if (fdx*fdx + fdy*fdy < gr2) {
                int old = atomicExch(&falive[i], 0);
                if (old) { energy = fminf(energy + 10.0f, 200.0f); score += 1.0f; }
            }
        }
    }
    scores[tid] = score;
}

int main() {
    printf("=== CONSTRAINT CONSTANT VERIFICATION v2 (Two-Phase Design) ===\n");
    printf("Grid,Agents_p1,Agents_p2,Food,Grab,Random,Away,Toward,Oracle,AwayBoost,AwayConst\n");
    
    int grids[] = {32, 64, 128, 256};
    int agents_p1[] = {64, 128, 256, 512};
    int agents_p2[] = {64, 128, 256, 512};
    int foods[] = {100, 200, 400, 800};
    float grabs[] = {3.0f, 5.0f, 7.0f};
    
    for (int gi = 0; gi < 4; gi++) {
        for (int a1i = 0; a1i < 4; a1i++) {
            for (int a2i = 0; a2i < 4; a2i++) {
                for (int fi = 0; fi < 4; fi++) {
                    for (int gri = 0; gri < 3; gri++) {
                        int grid_w = grids[gi];
                        float w = (float)grid_w * 4.0f; // world size
                        int n_p1 = agents_p1[a1i];
                        int n_p2 = agents_p2[a2i];
                        int food_count = foods[fi];
                        float grab_r = grabs[gri];
                        
                        if (n_p2 > MAX_AGENTS) continue;
                        if (food_count > MAX_FOOD) continue;
                        
                        int *d_tr, *d_fa;
                        float *d_s, *d_fx, *d_fy;
                        
                        float avg_scores[4] = {0, 0, 0, 0}; // random, toward, away, oracle
                        
                        for (int trial = 0; trial < TRIALS; trial++) {
                            // Allocate
                            cudaMalloc(&d_tr, grid_w * grid_w * sizeof(int));
                            cudaMalloc(&d_s, n_p2 * sizeof(float));
                            cudaMalloc(&d_fx, food_count * sizeof(float));
                            cudaMalloc(&d_fy, food_count * sizeof(float));
                            cudaMalloc(&d_fa, food_count * sizeof(int));
                            
                            // Place food
                            float hfx[MAX_FOOD], hfy[MAX_FOOD];
                            srand(42 + trial * 777);
                            for (int i = 0; i < food_count; i++) {
                                hfx[i] = ((float)rand() / RAND_MAX) * w;
                                hfy[i] = ((float)rand() / RAND_MAX) * w;
                            }
                            cudaMemcpy(d_fx, hfx, food_count * sizeof(float), cudaMemcpyHostToDevice);
                            cudaMemcpy(d_fy, hfy, food_count * sizeof(float), cudaMemcpyHostToDevice);
                            
                            int blk_p1 = (n_p1 + BLK - 1) / BLK;
                            int blk_p2 = (n_p2 + BLK - 1) / BLK;
                            
                            for (int mode = 0; mode < 4; mode++) {
                                // Reset traces
                                cudaMemset(d_tr, 0, grid_w * grid_w * sizeof(int));
                                
                                // Reset food alive
                                int ones[MAX_FOOD];
                                for (int i = 0; i < food_count; i++) ones[i] = 1;
                                cudaMemcpy(d_fa, ones, food_count * sizeof(int), cudaMemcpyHostToDevice);
                                
                                // Phase 1: Build traces (except for mode 0 random and mode 3 oracle)
                                if (mode == 1 || mode == 2) {
                                    phase1_traces<<<blk_p1, BLK>>>(d_tr, PHASE_STEPS, n_p1, w, grid_w, 
                                        (unsigned int)(42 + trial * 1111));
                                }
                                
                                // Phase 2: Exploit
                                phase2_exploit<<<blk_p2, BLK>>>(d_s, d_tr, d_fx, d_fy, d_fa,
                                    PHASE_STEPS, n_p2, food_count, w, grid_w, mode, grab_r,
                                    (unsigned int)(42 + trial * 1111));
                                cudaDeviceSynchronize();
                                
                                // Sum scores
                                float h_s[MAX_AGENTS];
                                cudaMemcpy(h_s, d_s, n_p2 * sizeof(float), cudaMemcpyDeviceToHost);
                                float total = 0;
                                for (int i = 0; i < n_p2; i++) total += h_s[i];
                                avg_scores[mode] += total / (float)n_p2;
                            }
                            
                            cudaFree(d_tr); cudaFree(d_s);
                            cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fa);
                        }
                        
                        for (int m = 0; m < 4; m++) avg_scores[m] /= TRIALS;
                        
                        float away_boost = (avg_scores[2] - avg_scores[0]) / avg_scores[0];
                        float away_const = (avg_scores[2] - avg_scores[0]) / (avg_scores[3] - avg_scores[0]);
                        float toward_boost = (avg_scores[1] - avg_scores[0]) / avg_scores[0];
                        
                        printf("%d,%d,%d,%d,%.1f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f\n",
                            grid_w, n_p1, n_p2, food_count, grab_r,
                            avg_scores[0], avg_scores[2], avg_scores[1], avg_scores[3],
                            away_boost, away_const);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    printf("\n=== COMPLETE ===\n");
    return 0;
}
