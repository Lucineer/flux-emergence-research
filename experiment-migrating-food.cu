// experiment-migrating-food.cu — Do agents with spatial memory outperform random walkers
// when food follows predictable migration patterns?
//
// Hypothesis: Agents that remember past food locations and extrapolate movement
// will outperform random walkers when food migrates in patterns.
// Counter-hypothesis: Grab range dominates so much that memory is irrelevant (Law 1).

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

struct Config {
    int world_size;
    int num_agents;
    int num_food;
    int steps;
    float grab_range;
    float move_speed;
    float food_migrate_speed;  // 0=static, higher=faster migration
    int migration_pattern;     // 0=static, 1=linear, 2=circular, 3=random walk
    int memory_enabled;        // 0=none, 1=predict, 2=nearest-past
    float perception_range;
};

__device__ Config cfg;

__device__ float curand_uniform(int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (float)(*seed) / 0x7fffffff;
}

__device__ float dist2(float ax, float ay, float bx, float by) {
    float dx = ax - bx, dy = ay - by;
    return dx*dx + dy*dy;
}

__device__ float wrap(float v, float w) {
    float r = fmodf(v, w);
    return r < 0 ? r + w : r;
}

__device__ float wrap_dist(float a, float b, float w) {
    float d = fabsf(a - b);
    return fminf(d, w - d);
}

// Agent state
__device__ float ax[AGENTS], ay[AGENTS];
__device__ int a_seed[AGENTS];
__device__ int a_food[AGENTS];
__device__ float a_mem_x[AGENTS][MAX_MEM];
__device__ float a_mem_y[AGENTS][MAX_MEM];
__device__ int a_mem_idx[AGENTS]; // circular buffer index
__device__ int a_mem_count[AGENTS]; // how many memories stored

// Food state
__device__ float fx[FOOD], fy[FOOD];
__device__ float fx_vel[FOOD], fy_vel[FOOD]; // migration velocity
__device__ int f_seed[FOOD];
__device__ float f_circle_angle[FOOD]; // for circular pattern
__device__ float f_circle_radius[FOOD];
__device__ float f_circle_cx[FOOD], f_circle_cy[FOOD];

__global__ void init_agents(int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.num_agents) return;
    a_seed[i] = seed + i * 137;
    ax[i] = curand_uniform(&a_seed[i]) * cfg.world_size;
    ay[i] = curand_uniform(&a_seed[i]) * cfg.world_size;
    a_food[i] = 0;
    a_mem_idx[i] = 0;
    a_mem_count[i] = 0;
    for (int m = 0; m < MAX_MEM; m++) {
        a_mem_x[i][m] = 0;
        a_mem_y[i][m] = 0;
    }
}

__global__ void init_food(int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.num_food) return;
    f_seed[i] = seed + i * 997;
    fx[i] = curand_uniform(&f_seed[i]) * cfg.world_size;
    fy[i] = curand_uniform(&f_seed[i]) * cfg.world_size;
    fx_vel[i] = 0;
    fy_vel[i] = 0;
    f_circle_cx[i] = curand_uniform(&f_seed[i]) * cfg.world_size;
    f_circle_cy[i] = curand_uniform(&f_seed[i]) * cfg.world_size;
    f_circle_radius[i] = 100 + curand_uniform(&f_seed[i]) * 200;
    f_circle_angle[i] = curand_uniform(&f_seed[i]) * 6.2832f;
}

__global__ void migrate_food() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.num_food) return;
    
    if (cfg.migration_pattern == 0) return; // static
    
    float spd = cfg.food_migrate_speed;
    
    if (cfg.migration_pattern == 1) {
        // Linear migration — each food item drifts in a random direction
        if (fx_vel[i] == 0 && fy_vel[i] == 0) {
            float angle = curand_uniform(&f_seed[i]) * 6.2832f;
            fx_vel[i] = cosf(angle) * spd;
            fy_vel[i] = sinf(angle) * spd;
        }
        fx[i] = wrap(fx[i] + fx_vel[i], cfg.world_size);
        fy[i] = wrap(fy[i] + fy_vel[i], cfg.world_size);
        
    } else if (cfg.migration_pattern == 2) {
        // Circular — food orbits a center point
        f_circle_angle[i] += spd * 0.01f;
        fx[i] = wrap(f_circle_cx[i] + cosf(f_circle_angle[i]) * f_circle_radius[i], cfg.world_size);
        fy[i] = wrap(f_circle_cy[i] + sinf(f_circle_angle[i]) * f_circle_radius[i], cfg.world_size);
        
    } else if (cfg.migration_pattern == 3) {
        // Random walk migration
        fx[i] = wrap(fx[i] + (curand_uniform(&f_seed[i]) - 0.5f) * 2 * spd, cfg.world_size);
        fy[i] = wrap(fy[i] + (curand_uniform(&f_seed[i]) - 0.5f) * 2 * spd, cfg.world_size);
    }
}

__global__ void step_agents() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.num_agents) return;
    
    float best_d = 1e9;
    float best_fx = 0, best_fy = 0;
    int found = 0;
    
    // Find nearest food
    for (int j = 0; j < cfg.num_food; j++) {
        float dx = wrap_dist(ax[i], fx[j], cfg.world_size);
        float dy = wrap_dist(ay[i], fy[j], cfg.world_size);
        float d = dx*dx + dy*dy;
        if (d < best_d) {
            best_d = d;
            best_fx = fx[j];
            best_fy = fy[j];
        }
    }
    
    float move_x = 0, move_y = 0;
    
    if (cfg.memory_enabled == 0) {
        // No memory — move toward nearest visible food or random
        if (best_d < cfg.perception_range * cfg.perception_range) {
            float dx = wrap_dist(best_fx, ax[i], cfg.world_size);
            float dy = wrap_dist(best_fy, ay[i], cfg.world_size);
            float d = sqrtf(best_d);
            if (d > 0) { move_x = dx/d * cfg.move_speed; move_y = dy/d * cfg.move_speed; }
        } else {
            float angle = curand_uniform(&a_seed[i]) * 6.2832f;
            move_x = cosf(angle) * cfg.move_speed;
            move_y = sinf(angle) * cfg.move_speed;
        }
    } else if (cfg.memory_enabled == 1) {
        // Memory mode: extrapolate food migration from past locations
        // If we have >= 2 memories, predict where food will be
        float pred_x = 0, pred_y = 0;
        int has_prediction = 0;
        
        if (a_mem_count[i] >= 2) {
            // Use last 2 memories to extrapolate velocity
            int idx1 = (a_mem_idx[i] - 1 + MAX_MEM) % MAX_MEM;
            int idx2 = (a_mem_idx[i] - 2 + MAX_MEM) % MAX_MEM;
            float vx = a_mem_x[i][idx1] - a_mem_x[i][idx2];
            float vy = a_mem_y[i][idx1] - a_mem_y[i][idx2];
            pred_x = wrap(a_mem_x[i][idx1] + vx, cfg.world_size);
            pred_y = wrap(a_mem_y[i][idx1] + vy, cfg.world_size);
            has_prediction = 1;
        } else if (a_mem_count[i] >= 1) {
            int idx = (a_mem_idx[i] - 1 + MAX_MEM) % MAX_MEM;
            pred_x = a_mem_x[i][idx];
            pred_y = a_mem_y[i][idx];
            has_prediction = 1;
        }
        
        if (has_prediction) {
            float dx = wrap_dist(pred_x, ax[i], cfg.world_size);
            float dy = wrap_dist(pred_y, ay[i], cfg.world_size);
            float d = sqrtf(dx*dx + dy*dy);
            if (d > 0) { move_x = dx/d * cfg.move_speed; move_y = dy/d * cfg.move_speed; }
        } else {
            float angle = curand_uniform(&a_seed[i]) * 6.2832f;
            move_x = cosf(angle) * cfg.move_speed;
            move_y = sinf(angle) * cfg.move_speed;
        }
        
        // Store food location in memory
        if (best_d < cfg.perception_range * cfg.perception_range) {
            a_mem_x[i][a_mem_idx[i]] = best_fx;
            a_mem_y[i][a_mem_idx[i]] = best_fy;
            a_mem_idx[i] = (a_mem_idx[i] + 1) % MAX_MEM;
            if (a_mem_count[i] < MAX_MEM) a_mem_count[i]++;
        }
        
    } else if (cfg.memory_enabled == 2) {
        // Nearest-past: go to last known food location
        if (a_mem_count[i] >= 1) {
            int idx = (a_mem_idx[i] - 1 + MAX_MEM) % MAX_MEM;
            float dx = wrap_dist(a_mem_x[i][idx], ax[i], cfg.world_size);
            float dy = wrap_dist(a_mem_y[i][idx], ay[i], cfg.world_size);
            float d = sqrtf(dx*dx + dy*dy);
            if (d > 0) { move_x = dx/d * cfg.move_speed; move_y = dy/d * cfg.move_speed; }
        } else {
            float angle = curand_uniform(&a_seed[i]) * 6.2832f;
            move_x = cosf(angle) * cfg.move_speed;
            move_y = sinf(angle) * cfg.move_speed;
        }
        
        if (best_d < cfg.perception_range * cfg.perception_range) {
            a_mem_x[i][a_mem_idx[i]] = best_fx;
            a_mem_y[i][a_mem_idx[i]] = best_fy;
            a_mem_idx[i] = (a_mem_idx[i] + 1) % MAX_MEM;
            if (a_mem_count[i] < MAX_MEM) a_mem_count[i]++;
        }
    }
    
    ax[i] = wrap(ax[i] + move_x, cfg.world_size);
    ay[i] = wrap(ay[i] + move_y, cfg.world_size);
    
    // Collect food within grab range
    for (int j = 0; j < cfg.num_food; j++) {
        float dx = wrap_dist(ax[i], fx[j], cfg.world_size);
        float dy = wrap_dist(ay[i], fy[j], cfg.world_size);
        float d2 = dx*dx + dy*dy;
        if (d2 < cfg.grab_range * cfg.grab_range) {
            a_food[i]++;
            // Respawn food
            fx[j] = curand_uniform(&f_seed[j]) * cfg.world_size;
            fy[j] = curand_uniform(&f_seed[j]) * cfg.world_size;
            // Reset migration for respawned food
            if (cfg.migration_pattern == 1) {
                float angle = curand_uniform(&f_seed[j]) * 6.2832f;
                fx_vel[j] = cosf(angle) * cfg.food_migrate_speed;
                fy_vel[j] = sinf(angle) * cfg.food_migrate_speed;
            }
            f_circle_cx[j] = curand_uniform(&f_seed[j]) * cfg.world_size;
            f_circle_cy[j] = curand_uniform(&f_seed[j]) * cfg.world_size;
            f_circle_radius[j] = 100 + curand_uniform(&f_seed[j]) * 200;
            f_circle_angle[j] = curand_uniform(&f_seed[j]) * 6.2832f;
            break;
        }
    }
}

void run_experiment(const char* label, int pattern, float migrate_speed, int memory_mode, int seed) {
    Config h_cfg = {WORLD, AGENTS, FOOD, STEPS, 15.0f, 3.0f, migrate_speed, pattern, memory_mode, 50.0f};
    cudaMemcpyToSymbol(cfg, &h_cfg, sizeof(Config));
    
    int blocks = (AGENTS + BLOCK - 1) / BLOCK;
    int fblocks = (FOOD + BLOCK - 1) / BLOCK;
    
    init_agents<<<blocks, BLOCK>>>(seed);
    init_food<<<fblocks, BLOCK>>>(seed + 42);
    cudaDeviceSynchronize();
    
    for (int s = 0; s < STEPS; s++) {
        migrate_food<<<fblocks, BLOCK>>>();
        step_agents<<<blocks, BLOCK>>>();
        cudaDeviceSynchronize();
    }
    
    int total_food = 0;
    int* d_food;
    // Copy agent food counts
    int h_food[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(int) * AGENTS);
    for (int i = 0; i < AGENTS; i++) total_food += h_food[i];
    
    printf("  %-30s mem=%d pattern=%d speed=%.1f  total=%d  per_agent=%.1f\n",
           label, memory_mode, pattern, migrate_speed, total_food, (float)total_food/AGENTS);
}

int main() {
    printf("=== Migrating Food + Spatial Memory ===\n");
    printf("Law test: Does spatial prediction beat perception when food moves?\n\n");
    
    // Phase 1: Static food baseline
    printf("--- Phase 1: Static Food (baseline) ---\n");
    run_experiment("No memory, static food", 0, 0, 0, 42);
    run_experiment("Predict memory, static", 0, 0, 1, 42);
    run_experiment("Nearest-past, static", 0, 0, 2, 42);
    
    // Phase 2: Linear migration
    printf("\n--- Phase 2: Linear Migration ---\n");
    run_experiment("No memory, linear 1.0", 1, 1.0f, 0, 42);
    run_experiment("Predict, linear 1.0", 1, 1.0f, 1, 42);
    run_experiment("Nearest-past, linear 1.0", 1, 1.0f, 2, 42);
    run_experiment("No memory, linear 3.0", 1, 3.0f, 0, 42);
    run_experiment("Predict, linear 3.0", 1, 3.0f, 1, 42);
    run_experiment("Nearest-past, linear 3.0", 1, 3.0f, 2, 42);
    
    // Phase 3: Circular migration (predictable)
    printf("\n--- Phase 3: Circular Migration (predictable) ---\n");
    run_experiment("No memory, circular 1.0", 2, 1.0f, 0, 42);
    run_experiment("Predict, circular 1.0", 2, 1.0f, 1, 42);
    run_experiment("Nearest-past, circular 1.0", 2, 1.0f, 2, 42);
    run_experiment("No memory, circular 3.0", 2, 3.0f, 0, 42);
    run_experiment("Predict, circular 3.0", 2, 3.0f, 1, 42);
    run_experiment("Nearest-past, circular 3.0", 2, 3.0f, 2, 42);
    
    // Phase 4: Random walk migration (unpredictable)
    printf("\n--- Phase 4: Random Walk Migration (unpredictable) ---\n");
    run_experiment("No memory, random 1.0", 3, 1.0f, 0, 42);
    run_experiment("Predict, random 1.0", 3, 1.0f, 1, 42);
    run_experiment("Nearest-past, random 1.0", 3, 1.0f, 2, 42);
    run_experiment("No memory, random 3.0", 3, 3.0f, 0, 42);
    run_experiment("Predict, random 3.0", 3, 3.0f, 1, 42);
    run_experiment("Nearest-past, random 3.0", 3, 3.0f, 2, 42);
    
    printf("\n=== Analysis ===\n");
    printf("If prediction beats no-memory on circular/linear: spatial memory matters\n");
    printf("If prediction HURTS on random walk: memory of stochastic patterns is harmful\n");
    printf("If nearest-past always beats prediction: simpler memory > extrapolation\n");
    
    return 0;
}
