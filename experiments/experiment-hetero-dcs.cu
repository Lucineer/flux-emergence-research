// experiment-hetero-dcs.cu
// Heterogeneous DCS — Mixed specialist + generalist populations
//
// Previous DCS result: guild-only no-filter = 7.5x over control.
// But all agents were identical. What happens with mixed populations?
//
// Hypothesis from emergence research:
// - Specialist ratio 20% = best total fitness
// - Specialist ratio 90% = best specialist avg
// - Under DCS, generalists get 21.87x uplift individually
// - Question: does DCS shift the optimal specialist ratio?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GRID_SIZE        128
#define NUM_AGENTS       1024
#define NUM_RESOURCES    200
#define STEPS            3000
#define NUM_TRIALS       5
#define GUILD_SIZE       32
#define GUILD_RADIUS     40.0f

// Agent types
#define TYPE_GENERALIST  0
#define TYPE_SPECIALIST_A 1
#define TYPE_SPECIALIST_B 2

struct Agent {
    float x, y;
    int type;
    float energy;
    int collected;
    float grab_range;
    int guild;
    float skill_a, skill_b, skill_g;  // collection efficiency per resource type
};

struct Resource {
    float x, y;
    int type;  // 0=normal, 1=special_A, 2=special_B
    float value;
    int active;
};

__device__ int d_total_collected;
__device__ int d_spec_a_collected;
__device__ int d_spec_b_collected;
__device__ int d_gen_collected;

__device__ float lcg(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}

__global__ void reset_stats() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_total_collected = 0;
        d_spec_a_collected = 0;
        d_spec_b_collected = 0;
        d_gen_collected = 0;
    }
}

__global__ void init_agents(Agent *agents, int n, float spec_ratio, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned int s = seed + idx * 7919;
    agents[idx].x = lcg(&s) * GRID_SIZE;
    agents[idx].y = lcg(&s) * GRID_SIZE;
    agents[idx].energy = 100.0f;
    agents[idx].collected = 0;
    agents[idx].grab_range = 3.0f;
    agents[idx].guild = idx % (n / GUILD_SIZE);
    
    int num_specs = (int)(n * spec_ratio);
    int half_specs = num_specs / 2;
    
    if (idx < half_specs) {
        agents[idx].type = TYPE_SPECIALIST_A;
        agents[idx].skill_a = 1.0f;  // perfect at A resources
        agents[idx].skill_b = 0.2f;
        agents[idx].skill_g = 0.2f;
    } else if (idx < num_specs) {
        agents[idx].type = TYPE_SPECIALIST_B;
        agents[idx].skill_a = 0.2f;
        agents[idx].skill_b = 1.0f;  // perfect at B resources
        agents[idx].skill_g = 0.2f;
    } else {
        agents[idx].type = TYPE_GENERALIST;
        agents[idx].skill_a = 0.6f;  // decent at everything
        agents[idx].skill_b = 0.6f;
        agents[idx].skill_g = 0.6f;
    }
}

__global__ void init_resources(Resource *resources, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned int s = seed + idx * 65537 + 999;
    resources[idx].x = lcg(&s) * GRID_SIZE;
    resources[idx].y = lcg(&s) * GRID_SIZE;
    resources[idx].value = 1.0f;
    resources[idx].active = 1;
    
    // Resource distribution: 60% normal, 20% type A, 20% type B
    float r = lcg(&s);
    if (r < 0.6f) resources[idx].type = 0;
    else if (r < 0.8f) resources[idx].type = 1;
    else resources[idx].type = 2;
}

// DCS: share guild knowledge
__global__ void dcs_share(Agent *agents, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int my_guild = agents[idx].guild;
    float guild_best_a = 0, guild_best_b = 0, guild_best_g = 0;
    int guild_count = 0;
    
    // Check nearby guild members (simplified: iterate all same-guild)
    // In production, use spatial hash. For benchmark, brute force is fine.
    for (int i = 0; i < n && guild_count < GUILD_SIZE; i++) {
        if (agents[i].guild == my_guild) {
            guild_count++;
            // Share skill knowledge (guild effect)
            guild_best_a = fmaxf(guild_best_a, agents[i].skill_a);
            guild_best_b = fmaxf(guild_best_b, agents[i].skill_b);
            guild_best_g = fmaxf(guild_best_g, agents[i].skill_g);
        }
    }
    
    // Boost skills slightly from guild knowledge
    float boost = 0.1f;  // 10% boost from guild
    agents[idx].skill_a = fminf(1.0f, agents[idx].skill_a + boost * 0.3f);
    agents[idx].skill_b = fminf(1.0f, agents[idx].skill_b + boost * 0.3f);
    agents[idx].skill_g = fminf(1.0f, agents[idx].skill_g + boost * 0.3f);
}

__global__ void simulate_step(
    Agent *agents, int n_agents,
    Resource *resources, int n_resources,
    int step, unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_agents) return;
    
    unsigned int s = seed + idx * 131 + step * 997;
    
    // Move randomly
    agents[idx].x += (lcg(&s) - 0.5f) * 4.0f;
    agents[idx].y += (lcg(&s) - 0.5f) * 4.0f;
    
    // Wrap
    agents[idx].x = fmodf(agents[idx].x + GRID_SIZE, GRID_SIZE);
    agents[idx].y = fmodf(agents[idx].y + GRID_SIZE, GRID_SIZE);
    
    // Try to collect resources
    float grab2 = agents[idx].grab_range * agents[idx].grab_range;
    
    for (int r = 0; r < n_resources; r++) {
        if (!resources[r].active) continue;
        
        float dx = agents[idx].x - resources[r].x;
        float dy = agents[idx].y - resources[r].y;
        
        // Toroidal distance
        if (dx > GRID_SIZE * 0.5f) dx -= GRID_SIZE;
        if (dx < -GRID_SIZE * 0.5f) dx += GRID_SIZE;
        if (dy > GRID_SIZE * 0.5f) dy -= GRID_SIZE;
        if (dy < -GRID_SIZE * 0.5f) dy += GRID_SIZE;
        
        float d2 = dx * dx + dy * dy;
        
        if (d2 < grab2) {
            // Check skill match
            float skill;
            if (resources[r].type == 1) skill = agents[idx].skill_a;
            else if (resources[r].type == 2) skill = agents[idx].skill_b;
            else skill = agents[idx].skill_g;
            
            // Collect with probability based on skill
            if (lcg(&s) < skill) {
                resources[r].active = 0;
                agents[idx].collected++;
                agents[idx].energy += resources[r].value;
                
                atomicAdd(&d_total_collected, 1);
                if (agents[idx].type == TYPE_SPECIALIST_A)
                    atomicAdd(&d_spec_a_collected, 1);
                else if (agents[idx].type == TYPE_SPECIALIST_B)
                    atomicAdd(&d_spec_b_collected, 1);
                else
                    atomicAdd(&d_gen_collected, 1);
            }
        }
    }
}

__global__ void respawn_resources(Resource *resources, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (!resources[idx].active) {
        unsigned int s = seed + idx * 3571;
        resources[idx].active = 1;
        resources[idx].x = lcg(&s) * GRID_SIZE;
        resources[idx].y = lcg(&s) * GRID_SIZE;
    }
}

int main() {
    printf("=== Heterogeneous DCS — Mixed Specialist/Generalist Populations ===\n");
    printf("Agents: %d | Resources: %d | Steps: %d | Trials: %d\n", 
           NUM_AGENTS, NUM_RESOURCES, STEPS, NUM_TRIALS);
    printf("Resource mix: 60%% normal, 20%% type A, 20%% type B\n\n");
    
    int blockSize = 256;
    int gridSize = (NUM_AGENTS + blockSize - 1) / blockSize;
    int resGridSize = (NUM_RESOURCES + blockSize - 1) / blockSize;
    
    Agent *d_agents;
    Resource *d_resources;
    cudaMalloc(&d_agents, NUM_AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, NUM_RESOURCES * sizeof(Resource));
    
    srand(time(NULL));
    
    // Sweep specialist ratios
    float spec_ratios[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f, 0.9f, 1.0f};
    int num_ratios = 9;
    
    printf("=== Without DCS (Control) ===\n");
    printf("%-12s %-10s %-10s %-10s %-10s %-10s\n",
           "Spec Ratio", "Total", "Spec A", "Spec B", "General", "Gen Avg");
    printf("%s\n", "------------------------------------------------------------");
    
    for (int r = 0; r < num_ratios; r++) {
        float ratio = spec_ratios[r];
        int total = 0, spec_a = 0, spec_b = 0, gen = 0;
        
        for (int t = 0; t < NUM_TRIALS; t++) {
            unsigned int seed = (unsigned int)time(NULL) + t * 10000 + r * 100000;
            
            reset_stats<<<1, 1>>>();
            init_agents<<<gridSize, blockSize>>>(d_agents, NUM_AGENTS, ratio, seed);
            init_resources<<<resGridSize, blockSize>>>(d_resources, NUM_RESOURCES, seed + 1);
            cudaDeviceSynchronize();
            
            for (int step = 0; step < STEPS; step++) {
                simulate_step<<<gridSize, blockSize>>>(d_agents, NUM_AGENTS, d_resources, NUM_RESOURCES, step, seed);
                respawn_resources<<<resGridSize, blockSize>>>(d_resources, NUM_RESOURCES, seed + step);
                
                if (step % 100 == 0) {
                    // Reset skills to prevent accumulation (no DCS)
                    init_agents<<<gridSize, blockSize>>>(d_agents, NUM_AGENTS, ratio, seed);
                }
                
                cudaDeviceSynchronize();
            }
            
            int h_total, h_sa, h_sb, h_gen;
            cudaMemcpyFromSymbol(&h_total, d_total_collected, sizeof(int));
            cudaMemcpyFromSymbol(&h_sa, d_spec_a_collected, sizeof(int));
            cudaMemcpyFromSymbol(&h_sb, d_spec_b_collected, sizeof(int));
            cudaMemcpyFromSymbol(&h_gen, d_gen_collected, sizeof(int));
            total += h_total; spec_a += h_sa; spec_b += h_sb; gen += h_gen;
        }
        
        int num_gen = (int)(NUM_AGENTS * (1.0f - ratio));
        float gen_avg = num_gen > 0 ? (float)gen / (NUM_TRIALS * num_gen) : 0;
        
        printf("%-12.1f %-10d %-10d %-10d %-10d %-10.2f\n",
               ratio, total / NUM_TRIALS, spec_a / NUM_TRIALS,
               spec_b / NUM_TRIALS, gen / NUM_TRIALS, gen_avg);
    }
    
    printf("\n=== With DCS (Guild sharing, every 50 steps) ===\n");
    printf("%-12s %-10s %-10s %-10s %-10s %-10s %-10s\n",
           "Spec Ratio", "Total", "Spec A", "Spec B", "General", "Gen Avg", "DCS Lift");
    printf("%s\n", "----------------------------------------------------------------------");
    
    // Store control results for comparison
    int control_totals[9];
    int idx = 0;
    
    for (int r = 0; r < num_ratios; r++) {
        float ratio = spec_ratios[r];
        int total = 0, spec_a = 0, spec_b = 0, gen = 0;
        
        for (int t = 0; t < NUM_TRIALS; t++) {
            unsigned int seed = (unsigned int)time(NULL) + t * 10000 + r * 100000 + 999;
            
            reset_stats<<<1, 1>>>();
            init_agents<<<gridSize, blockSize>>>(d_agents, NUM_AGENTS, ratio, seed);
            init_resources<<<resGridSize, blockSize>>>(d_resources, NUM_RESOURCES, seed + 1);
            cudaDeviceSynchronize();
            
            for (int step = 0; step < STEPS; step++) {
                simulate_step<<<gridSize, blockSize>>>(d_agents, NUM_AGENTS, d_resources, NUM_RESOURCES, step, seed);
                respawn_resources<<<resGridSize, blockSize>>>(d_resources, NUM_RESOURCES, seed + step);
                
                // DCS: share guild knowledge every 50 steps
                if (step % 50 == 0 && step > 0) {
                    dcs_share<<<gridSize, blockSize>>>(d_agents, NUM_AGENTS);
                }
                
                cudaDeviceSynchronize();
            }
            
            int h_total, h_sa, h_sb, h_gen;
            cudaMemcpyFromSymbol(&h_total, d_total_collected, sizeof(int));
            cudaMemcpyFromSymbol(&h_sa, d_spec_a_collected, sizeof(int));
            cudaMemcpyFromSymbol(&h_sb, d_spec_b_collected, sizeof(int));
            cudaMemcpyFromSymbol(&h_gen, d_gen_collected, sizeof(int));
            total += h_total; spec_a += h_sa; spec_b += h_sb; gen += h_gen;
        }
        
        int num_gen = (int)(NUM_AGENTS * (1.0f - ratio));
        float gen_avg = num_gen > 0 ? (float)gen / (NUM_TRIALS * num_gen) : 0;
        float dcs_lift = control_totals[idx] > 0 ? (float)total / (NUM_TRIALS * control_totals[idx]) : 0;
        control_totals[idx] = total / NUM_TRIALS;
        
        printf("%-12.1f %-10d %-10d %-10d %-10d %-10.2f %-10.2fx\n",
               ratio, total / NUM_TRIALS, spec_a / NUM_TRIALS,
               spec_b / NUM_TRIALS, gen / NUM_TRIALS, gen_avg, dcs_lift);
    }
    
    printf("\n=== Key Question: Does DCS shift the optimal specialist ratio? ===\n");
    printf("Control: peak at ratio 0.2 (from prior research)\n");
    printf("DCS: if peak shifts, protocol changes population dynamics.\n");
    
    cudaFree(d_agents);
    cudaFree(d_resources);
    
    return 0;
}
