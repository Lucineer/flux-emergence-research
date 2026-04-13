
/*
CUDA Simulation Experiment v88: Stigmergy + Memory
Testing: Agents leave pheromone trails at resource locations that decay over time.
         Agents have memory of recent pheromone locations (weighted average).
Prediction: Stigmergy will REDUCE specialist advantage because:
            1. Uniform agents benefit more from shared information trails
            2. Memory creates indirect coordination, reducing need for specialization
            3. Pheromone trails homogenize search strategies
Expected: Specialization ratio < 1.61x (v8 baseline), possibly near 1.0x
Baseline: v8 mechanisms (scarcity, territory, comms) included
Novel: Pheromone trails + agent memory
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define AGENTS 1024
#define RESOURCES 128
#define TICKS 500
#define ARCHETYPES 4
#define PHEROMONE_GRID 64  // 64x64 grid for pheromone map
#define MEMORY_DECAY 0.95f

// Linear Congruential Generator
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

struct Resource {
    float x, y;
    float value;
    bool collected;
    unsigned int spawn_timer;
};

struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPES];  // explore, collect, communicate, defend
    float fitness;
    int arch;  // archetype 0-3
    unsigned int rng;
    // Memory for pheromone guidance
    float mem_phero_x;
    float mem_phero_y;
    float mem_strength;
};

// Pheromone grid for stigmergy
__device__ float pheromone[PHEROMONE_GRID][PHEROMONE_GRID];

__global__ void init_agents(Agent *agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    unsigned int seed = idx * 17 + 12345;
    agents[idx].x = lcgf(seed) * 2.0f - 1.0f;
    agents[idx].y = lcgf(seed) * 2.0f - 1.0f;
    agents[idx].vx = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].rng = idx * 19 + 67890;
    agents[idx].arch = idx % ARCHETYPES;
    
    // Memory initialization
    agents[idx].mem_phero_x = 0.0f;
    agents[idx].mem_phero_y = 0.0f;
    agents[idx].mem_strength = 0.0f;
    
    if (specialized) {
        // Specialized: strong in one role (0.7), weak in others (0.1)
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = 0.1f;
        }
        agents[idx].role[agents[idx].arch] = 0.7f;
    } else {
        // Uniform: all roles equal
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

__global__ void init_resources(Resource *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int seed = idx * 23 + 54321;
    resources[idx].x = lcgf(seed) * 2.0f - 1.0f;
    resources[idx].y = lcgf(seed) * 2.0f - 1.0f;
    resources[idx].value = 0.5f + lcgf(seed) * 0.5f;
    resources[idx].collected = false;
    resources[idx].spawn_timer = 0;
}

__global__ void clear_pheromone() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < PHEROMONE_GRID && y < PHEROMONE_GRID) {
        // Exponential decay of existing pheromone
        pheromone[x][y] *= 0.8f;
        if (pheromone[x][y] < 0.001f) pheromone[x][y] = 0.0f;
    }
}

__device__ void deposit_pheromone(float x, float y, float amount) {
    int grid_x = (int)((x + 1.0f) * 0.5f * (PHEROMONE_GRID - 1));
    int grid_y = (int)((y + 1.0f) * 0.5f * (PHEROMONE_GRID - 1));
    
    if (grid_x >= 0 && grid_x < PHEROMONE_GRID && 
        grid_y >= 0 && grid_y < PHEROMONE_GRID) {
        atomicAdd(&pheromone[grid_x][grid_y], amount);
    }
}

__device__ float sample_pheromone(float x, float y) {
    int grid_x = (int)((x + 1.0f) * 0.5f * (PHEROMONE_GRID - 1));
    int grid_y = (int)((y + 1.0f) * 0.5f * (PHEROMONE_GRID - 1));
    
    if (grid_x >= 0 && grid_x < PHEROMONE_GRID && 
        grid_y >= 0 && grid_y < PHEROMONE_GRID) {
        return pheromone[grid_x][grid_y];
    }
    return 0.0f;
}

__global__ void tick(Agent *agents, Resource *resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: prevent role homogenization
    float similarity = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) {
        similarity += a.role[i] * a.role[i];
    }
    similarity = sqrtf(similarity);
    
    if (similarity > 0.9f) {
        // Find non-dominant role
        int non_dom = 0;
        for (int i = 1; i < ARCHETYPES; i++) {
            if (a.role[i] < a.role[non_dom]) non_dom = i;
        }
        // Apply random drift
        a.role[non_dom] += lcgf(a.rng) * 0.02f - 0.01f;
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < ARCHETYPES; i++) sum += a.role[i];
        for (int i = 0; i < ARCHETYPES; i++) a.role[i] /= sum;
    }
    
    // Update memory with current pheromone gradient
    float current_phero = sample_pheromone(a.x, a.y);
    if (current_phero > a.mem_strength) {
        a.mem_phero_x = a.x;
        a.mem_phero_y = a.y;
        a.mem_strength = current_phero;
    }
    // Memory decay
    a.mem_strength *= MEMORY_DECAY;
    
    // Movement influenced by memory and roles
    float target_x = 0.0f, target_y = 0.0f;
    float explore_weight = a.role[0];  // explore role
    
    if (a.mem_strength > 0.1f && lcgf(a.rng) < 0.3f) {
        // Follow memory of pheromone trail
        target_x = a.mem_phero_x - a.x;
        target_y = a.mem_phero_y - a.y;
    } else {
        // Random exploration weighted by explore role
        target_x = (lcgf(a.rng) * 2.0f - 1.0f) * explore_weight;
        target_y = (lcgf(a.rng) * 2.0f - 1.0f) * explore_weight;
    }
    
    // Normalize and apply velocity
    float dist = sqrtf(target_x * target_x + target_y * target_y);
    if (dist > 0.0f) {
        target_x /= dist;
        target_y /= dist;
    }
    
    a.vx = a.vx * 0.7f + target_x * 0.03f;
    a.vy = a.vy * 0.7f + target_y * 0.03f;
    
    // Boundary check
    if (a.x + a.vx < -1.0f || a.x + a.vx > 1.0f) a.vx = -a.vx;
    if (a.y + a.vy < -1.0f || a.y + a.vy > 1.0f) a.vy = -a.vy;
    
    a.x += a.vx;
    a.y += a.vy;
    
    // Resource interaction
    float detect_range = 0.03f + a.role[0] * 0.04f;  // explore role increases detection
    float grab_range = 0.02f + a.role[1] * 0.02f;    // collect role increases grab
    
    // Find nearest resource
    int nearest_idx = -1;
    float nearest_dist = 1e6f;
    
    for (int r = 0; r < RESOURCES; r++) {
        Resource &res = resources[r];
        if (res.collected && res.spawn_timer > 0) {
            res.spawn_timer--;
            if (res.spawn_timer == 0) {
                res.collected = false;
            }
            continue;
        }
        
        if (!res.collected) {
            float dx = res.x - a.x;
            float dy = res.y - a.y;
            float dist2 = dx * dx + dy * dy;
            
            if (dist2 < nearest_dist) {
                nearest_dist = dist2;
                nearest_idx = r;
            }
        }
    }
    
    // Communication role: broadcast location if resource found
    if (nearest_idx != -1 && sqrtf(nearest_dist) < detect_range) {
        Resource &res = resources[nearest_idx];
        
        // Deposit pheromone at resource location (NOVEL MECHANISM)
        deposit_pheromone(res.x, res.y, 0.5f + a.role[2] * 0.5f);
        
        // Communication to nearby agents
        float comm_range = 0.06f;
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent &other = agents[j];
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx * dx + dy * dy < comm_range * comm_range) {
                // Influence other agent's movement toward resource
                if (lcgf(a.rng) < a.role[2]) {  // communication role probability
                    other.vx += (res.x - other.x) * 0.01f;
                    other.vy += (res.y - other.y) * 0.01f;
                }
            }
        }
        
        // Collection if in range
        if (sqrtf(nearest_dist) < grab_range) {
            float base_value = res.value;
            float collect_bonus = 1.0f + a.role[1] * 0.5f;  // collect role bonus
            
            // Territory defense bonus
            float defense_bonus = 1.0f;
            int nearby_defenders = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent &other = agents[j];
                if (other.arch == a.arch) {
                    float dx = other.x - a.x;
                    float dy = other.y - a.y;
                    if (dx * dx + dy * dy < 0.04f) {
                        defense_bonus += other.role[3] * 0.2f;  // defend role contribution
                        nearby_defenders++;
                    }
                }
            }
            
            float total_value = base_value * collect_bonus * defense_bonus;
            a.energy += total_value;
            a.fitness += total_value;
            
            // Mark resource as collected
            res.collected = true;
            res.spawn_timer = 50;  // respawn timer
            
            // Deposit stronger pheromone when collecting (NOVEL)
            deposit_pheromone(res.x, res.y, 1.0f + a.role[1]);
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0 && lcgf(a.rng) < 0.1f) {
        // Defenders resist perturbation
        if (lcgf(a.rng) > a.role[3]) {  // defend role provides resistance
            a.energy *= 0.5f;
            a.vx = lcgf(a.rng) * 0.1f - 0.05f;
            a.vy = lcgf(a.rng) * 0.1f - 0.05f;
        }
    }
    
    // Coupling: role adaptation from nearby agents
    float coupling_same = 0.02f;
    float coupling_diff = 0.002f;
    
    for (int j = 0; j < AGENTS; j++) {
        if (j == idx) continue;
        
        Agent &other = agents[j];
        float dx = other.x - a.x;
        float dy = other.y - a.y;
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < 0.04f) {  // Interaction radius
            float coupling = (a.arch == other.arch) ? coupling_same : coupling_diff;
            
            for (int r = 0; r < ARCHETYPES; r++) {
                float delta = other.role[r] - a.role[r];
                a.role[r] += delta * coupling * lcgf(a.rng);
            }
        }
    }
    
    // Role normalization
    float sum = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) sum += a.role[i];
    if (sum > 0.0f) {
        for (int i = 0; i < ARCHETYPES; i++) a.role[i] /= sum;
    }
}

int main() {
    printf("Experiment v88: Stigmergy + Memory\n");
    printf("Testing: Pheromone trails + agent memory\n");
    printf("Prediction: Reduces specialist advantage (ratio < 1.61x)\n\n");
    
    // Allocate memory
    Agent *agents_spec, *agents_uniform;
    Resource *resources;
    
    cudaMallocManaged(&agents_spec, AGENTS * sizeof(Agent));
    cudaMallocManaged(&agents_uniform, AGENTS * sizeof(Agent));
    cudaMallocManaged(&resources, RESOURCES * sizeof(Resource));
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    
    init_agents<<<grid_spec, block>>>(agents_spec, 1);  // Specialized
    init_agents<<<grid_spec, block>>>(agents_uniform, 0);  // Uniform
    init_resources<<<grid_res, block>>>(resources);
    
    cudaDeviceSynchronize();
    
    // Initialize pheromone grid to zero
    cudaMemset2D(pheromone, PHEROMONE_GRID * sizeof(float), 0, PHEROMONE_GRID * sizeof(float), PHEROMONE_GRID);
    
    // Run simulation for specialized group
    printf("Running specialized group...\n");
    for (int t = 0; t < TICKS; t++) {
        clear_pheromone<<<dim3(8, 8), dim3(8, 8)>>>();
        tick<<<grid_spec, block>>>(agents_spec, resources, t);
        if (t % 100 == 0) {
            cudaDeviceSynchronize();
            printf("  Tick %d/500\n", t);
        }
    }
    cudaDeviceSynchronize();
    
    // Calculate average fitness for specialized
    float avg_fitness_spec = 0.0f;
    for (int i = 0; i < AGENTS; i++) {
        avg_fitness_spec += agents_spec[i].fitness;
    }
    avg_fitness_spec /= AGENTS;
    
    // Re-initialize resources for uniform group
    init_resources<<<