// experiment-constant-sweep.cu
// Law 267: Verify the coverage constant across parameter ranges
// Tests: grid size, agent count, food density, noise level
// Key question: Is ~0.415 stable, or does it converge to 1-1/e (0.632)?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_AGENTS 4096
#define MAX_FOOD 4096
#define MAX_GRID 512
#define STEPS 500
#define TRIALS 5

struct Params {
    int grid_size;
    int num_agents;
    int num_food;
    int noise_level;
    float grab_range;
};

__device__ int curand_state_arr[MAX_AGENTS * 4];
__device__ unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__global__ void run_simulation(
    int *d_food_x, int *d_food_y, int *d_noise_x, int *d_noise_y,
    int grid_size, int num_agents, int num_food, int noise_level,
    float grab_range, int steps, int trial, unsigned int seed,
    int *d_fitness_noisy, int *d_fitness_random, int *d_fitness_oracle
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_agents * 3) return;
    
    int agent_type = tid / num_agents; // 0=noisy, 1=random, 2=oracle
    int agent_id = tid % num_agents;
    
    unsigned int rng = seed + agent_id * 997 + agent_type * 1301 + trial * 7919;
    
    int ax = xorshift32(&rng) % grid_size;
    int ay = xorshift32(&rng) % grid_size;
    
    int total_food = 0;
    
    for (int s = 0; s < steps; s++) {
        // Move
        int dx = 0, dy = 0;
        if (agent_type == 0) {
            // NOISY: follow noise trace if near one, else random
            float best_dist = 999999.0f;
            int bx = -1, by = -1;
            for (int n = 0; n < noise_level; n++) {
                float ddx = (float)(d_noise_x[n] - ax);
                float ddy = (float)(d_noise_y[n] - ay);
                float d = sqrtf(ddx*ddx + ddy*ddy);
                if (d < best_dist) { best_dist = d; bx = d_noise_x[n]; by = d_noise_y[n]; }
            }
            if (best_dist < grab_range * 3.0f) {
                dx = (bx > ax) ? 1 : (bx < ax) ? -1 : 0;
                dy = (by > ay) ? 1 : (by < ay) ? -1 : 0;
            } else {
                dx = ((int)(xorshift32(&rng) % 3)) - 1;
                dy = ((int)(xorshift32(&rng) % 3)) - 1;
            }
        } else if (agent_type == 1) {
            // RANDOM: pure random walk
            dx = ((int)(xorshift32(&rng) % 3)) - 1;
            dy = ((int)(xorshift32(&rng) % 3)) - 1;
        } else {
            // ORACLE: move toward nearest food
            float best_dist = 999999.0f;
            int bx = ax, by = ay;
            for (int f = 0; f < num_food; f++) {
                float ddx = (float)(d_food_x[f] - ax);
                float ddy = (float)(d_food_y[f] - ay);
                float d = sqrtf(ddx*ddx + ddy*ddy);
                if (d < best_dist) { best_dist = d; bx = d_food_x[f]; by = d_food_y[f]; }
            }
            dx = (bx > ax) ? 1 : (bx < ax) ? -1 : 0;
            dy = (by > ay) ? 1 : (by < ay) ? -1 : 0;
        }
        
        ax = (ax + dx + grid_size) % grid_size;
        ay = (ay + dy + grid_size) % grid_size;
        
        // Collect food
        for (int f = 0; f < num_food; f++) {
            if (d_food_x[f] >= 0) {
                float ddx = (float)(d_food_x[f] - ax);
                float ddy = (float)(d_food_y[f] - ay);
                if (sqrtf(ddx*ddx + ddy*ddy) < grab_range) {
                    total_food++;
                    d_food_x[f] = -1; // consumed
                }
            }
        }
    }
    
    if (agent_type == 0) atomicAdd(d_fitness_noisy, total_food);
    else if (agent_type == 1) atomicAdd(d_fitness_random, total_food);
    else atomicAdd(d_fitness_oracle, total_food);
}

int main() {
    printf("=== CONSTRAINT CONSTANT VERIFICATION (Law 267) ===\n");
    printf("Grid,Agents,Food,Noise,Grab,Random,Noisy,Oracle,NoisyBoost,Const\n");
    
    // Parameter sweeps
    int grids[] = {32, 64, 128, 256};
    int agents_list[] = {64, 128, 256, 512};
    int food_list[] = {100, 200, 400, 800};
    float grabs[] = {2.0f, 4.0f, 6.0f};
    
    int total_configs = 4 * 4 * 4 * 3;
    int config_idx = 0;
    
    for (int gi = 0; gi < 4; gi++) {
        for (int ai = 0; ai < 4; ai++) {
            for (int fi = 0; fi < 4; fi++) {
                for (int gri = 0; gri < 3; gri++) {
                    int grid_size = grids[gi];
                    int num_agents = agents_list[ai];
                    int num_food = food_list[fi];
                    float grab_range = grabs[gri];
                    int noise_level = num_food; // noise = same count as food
                    
                    if (num_agents > grid_size * grid_size / 4) continue;
                    if (num_food > grid_size * grid_size / 4) continue;
                    
                    // Allocate
                    int *h_food_x = (int*)malloc(num_food * sizeof(int));
                    int *h_food_y = (int*)malloc(num_food * sizeof(int));
                    int *h_noise_x = (int*)malloc(noise_level * sizeof(int));
                    int *h_noise_y = (int*)malloc(noise_level * sizeof(int));
                    
                    int *d_food_x, *d_food_y, *d_noise_x, *d_noise_y;
                    int *d_fitness_noisy, *d_fitness_random, *d_fitness_oracle;
                    
                    float avg_noisy = 0, avg_random = 0, avg_oracle = 0;
                    
                    for (int trial = 0; trial < TRIALS; trial++) {
                        // Place food randomly
                        for (int i = 0; i < num_food; i++) {
                            h_food_x[i] = rand() % grid_size;
                            h_food_y[i] = rand() % grid_size;
                        }
                        // Place noise traces randomly (ZERO correlation to food)
                        for (int i = 0; i < noise_level; i++) {
                            h_noise_x[i] = rand() % grid_size;
                            h_noise_y[i] = rand() % grid_size;
                        }
                        
                        cudaMalloc(&d_food_x, num_food * sizeof(int));
                        cudaMalloc(&d_food_y, num_food * sizeof(int));
                        cudaMalloc(&d_noise_x, noise_level * sizeof(int));
                        cudaMalloc(&d_noise_y, noise_level * sizeof(int));
                        cudaMalloc(&d_fitness_noisy, sizeof(int));
                        cudaMalloc(&d_fitness_random, sizeof(int));
                        cudaMalloc(&d_fitness_oracle, sizeof(int));
                        
                        cudaMemcpy(d_food_x, h_food_x, num_food * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_food_y, h_food_y, num_food * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_noise_x, h_noise_x, noise_level * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_noise_y, h_noise_y, noise_level * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemset(d_fitness_noisy, 0, sizeof(int));
                        cudaMemset(d_fitness_random, 0, sizeof(int));
                        cudaMemset(d_fitness_oracle, 0, sizeof(int));
                        
                        int total_threads = num_agents * 3;
                        int threads = 256;
                        int blocks = (total_threads + threads - 1) / threads;
                        
                        run_simulation<<<blocks, threads>>>(
                            d_food_x, d_food_y, d_noise_x, d_noise_y,
                            grid_size, num_agents, num_food, noise_level,
                            grab_range, STEPS, trial, rand(),
                            d_fitness_noisy, d_fitness_random, d_fitness_oracle
                        );
                        cudaDeviceSynchronize();
                        
                        int h_noisy, h_random, h_oracle;
                        cudaMemcpy(&h_noisy, d_fitness_noisy, sizeof(int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_random, d_fitness_random, sizeof(int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_oracle, d_fitness_oracle, sizeof(int), cudaMemcpyDeviceToHost);
                        
                        avg_noisy += (float)h_noisy / (float)num_agents;
                        avg_random += (float)h_random / (float)num_agents;
                        avg_oracle += (float)h_oracle / (float)num_agents;
                        
                        cudaFree(d_food_x); cudaFree(d_food_y);
                        cudaFree(d_noise_x); cudaFree(d_noise_y);
                        cudaFree(d_fitness_noisy); cudaFree(d_fitness_random); cudaFree(d_fitness_oracle);
                    }
                    
                    avg_noisy /= TRIALS;
                    avg_random /= TRIALS;
                    avg_oracle /= TRIALS;
                    
                    float boost = (avg_noisy - avg_random) / avg_random;
                    float const_val = (avg_noisy - avg_random) / (avg_oracle - avg_random);
                    
                    printf("%d,%d,%d,%d,%.1f,%.3f,%.3f,%.3f,%.4f,%.4f\n",
                        grid_size, num_agents, num_food, noise_level, grab_range,
                        avg_random, avg_noisy, avg_oracle, boost, const_val);
                    
                    config_idx++;
                    fflush(stdout);
                    
                    free(h_food_x); free(h_food_y);
                    free(h_noise_x); free(h_noise_y);
                }
            }
        }
    }
    
    printf("\n=== COMPLETE: %d configurations tested ===\n", config_idx);
    return 0;
}
