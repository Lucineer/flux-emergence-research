// CUDA Simulation Experiment v54: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone trails at resource locations that decay over time
// Prediction: Pheromones will enhance specialist coordination, increasing advantage ratio >1.61x
// Novelty: Stigmergy (indirect communication through environment modification)
// Baseline: v8 mechanisms (scarcity, territory, comms) included
// Control: Uniform agents (all roles=0.25) vs Specialized (role[arch]=0.7)

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;
const int PHEROMONE_GRID = 256; // 256x256 grid
const float WORLD_SIZE = 1.0f;
const float CELL_SIZE = WORLD_SIZE / PHEROMONE_GRID;

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Agent struct
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Archetype 0-3
    unsigned int rng;     // RNG state
    float memory_x;       // Remembered resource location
    float memory_y;
    int memory_valid;     // Is memory valid?
};

// Resource struct
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection status
    int pheromone_strength; // Pheromone deposit counter
    unsigned int rng;     // RNG state for respawn
};

// Pheromone grid (global memory)
__device__ float pheromone_grid[PHEROMONE_GRID][PHEROMONE_GRID];
__device__ float pheromone_decay = 0.95f; // Decay per tick

// Initialize pheromone grid
__global__ void init_pheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < PHEROMONE_GRID * PHEROMONE_GRID; i += stride) {
        int x = i % PHEROMONE_GRID;
        int y = i / PHEROMONE_GRID;
        pheromone_grid[y][x] = 0.0f;
    }
}

// Decay pheromones
__global__ void decay_pheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < PHEROMONE_GRID * PHEROMONE_GRID; i += stride) {
        int x = i % PHEROMONE_GRID;
        int y = i / PHEROMONE_GRID;
        pheromone_grid[y][x] *= pheromone_decay;
        if (pheromone_grid[y][x] < 0.001f) pheromone_grid[y][x] = 0.0f;
    }
}

// Initialize agents
__global__ void init_agents(Agent* agents, int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    a.rng = idx * 17 + 12345;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = (lcgf(a.rng) - 0.5f) * 0.02f;
    a.vy = (lcgf(a.rng) - 0.5f) * 0.02f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % 4;
    a.memory_valid = 0;
    
    if (specialized) {
        // Specialized agents: strong in one role based on archetype
        for (int i = 0; i < 4; i++) a.role[i] = 0.1f;
        a.role[a.arch] = 0.7f;
    } else {
        // Uniform control: all roles equal
        for (int i = 0; i < 4; i++) a.role[i] = 0.25f;
    }
}

// Initialize resources
__global__ void init_resources(Resource* resources) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    Resource& r = resources[idx];
    r.rng = idx * 13 + 67890;
    r.x = lcgf(r.rng);
    r.y = lcgf(r.rng);
    r.value = 0.5f + lcgf(r.rng) * 0.5f;
    r.collected = 0;
    r.pheromone_strength = 0;
}

// Get pheromone value at position
__device__ float get_pheromone(float x, float y) {
    int gx = min(max((int)(x / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    int gy = min(max((int)(y / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    return pheromone_grid[gy][gx];
}

// Add pheromone at position
__device__ void add_pheromone(float x, float y, float amount) {
    int gx = min(max((int)(x / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    int gy = min(max((int)(y / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    atomicAdd(&pheromone_grid[gy][gx], amount);
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    if (a.energy <= 0.0f) return;
    
    // Anti-convergence: check similarity with nearby agents
    int similar_count = 0;
    int total_count = 0;
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent& other = agents[i];
        float dx = a.x - other.x;
        float dy = a.y - other.y;
        if (dx*dx + dy*dy < 0.01f) { // Within 0.1 distance
            total_count++;
            float similarity = 0.0f;
            for (int r = 0; r < 4; r++) {
                similarity += fabs(a.role[r] - other.role[r]);
            }
            if (similarity < 0.4f) similar_count++; // Similar if total diff < 0.4
        }
    }
    
    // Apply anti-convergence drift if too similar
    if (total_count > 3 && similar_count > total_count * 0.9f) {
        int drift_role = (int)(lcgf(a.rng) * 4) % 4;
        if (drift_role != a.arch) { // Don't drift dominant role
            a.role[drift_role] += (lcgf(a.rng) - 0.5f) * 0.02f;
            a.role[drift_role] = max(0.0f, min(1.0f, a.role[drift_role]));
        }
    }
    
    // Normalize roles
    float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    for (int i = 0; i < 4; i++) a.role[i] /= sum;
    
    // Role-based behavior
    float explore_strength = a.role[0];
    float collect_strength = a.role[1];
    float comm_strength = a.role[2];
    float defend_strength = a.role[3];
    
    // Pheromone sensing (NOVEL MECHANISM)
    float current_pheromone = get_pheromone(a.x, a.y);
    float pheromone_influence = 0.0f;
    
    // Sample pheromone in 4 directions
    float pheromones[4] = {0};
    pheromones[0] = get_pheromone(a.x + 0.01f, a.y);
    pheromones[1] = get_pheromone(a.x - 0.01f, a.y);
    pheromones[2] = get_pheromone(a.x, a.y + 0.01f);
    pheromones[3] = get_pheromone(a.x, a.y - 0.01f);
    
    // Move toward strongest pheromone if exploring
    if (explore_strength > 0.3f && current_pheromone < 0.1f) {
        int max_dir = 0;
        float max_val = pheromones[0];
        for (int i = 1; i < 4; i++) {
            if (pheromones[i] > max_val) {
                max_val = pheromones[i];
                max_dir = i;
            }
        }
        
        if (max_val > 0.05f) {
            switch (max_dir) {
                case 0: a.vx += 0.001f * explore_strength; break;
                case 1: a.vx -= 0.001f * explore_strength; break;
                case 2: a.vy += 0.001f * explore_strength; break;
                case 3: a.vy -= 0.001f * explore_strength; break;
            }
        }
    }
    
    // Exploration movement (random walk biased by pheromones)
    a.vx += (lcgf(a.rng) - 0.5f) * 0.002f * explore_strength;
    a.vy += (lcgf(a.rng) - 0.5f) * 0.002f * explore_strength;
    
    // Velocity damping and bounds
    a.vx *= 0.95f;
    a.vy *= 0.95f;
    a.x += a.vx;
    a.y += a.vy;
    
    // World bounds
    if (a.x < 0.0f) { a.x = 0.0f; a.vx = fabs(a.vx); }
    if (a.x > 1.0f) { a.x = 1.0f; a.vx = -fabs(a.vx); }
    if (a.y < 0.0f) { a.y = 0.0f; a.vy = fabs(a.vy); }
    if (a.y > 1.0f) { a.y = 1.0f; a.vy = -fabs(a.vy); }
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_idx = -1;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource& r = resources[i];
        if (r.collected) continue;
        
        float dx = a.x - r.x;
        float dy = a.y - r.y;
        float dist = dx*dx + dy*dy;
        
        // Detection range based on explore role
        float detect_range = 0.03f + 0.04f * explore_strength;
        
        if (dist < detect_range*detect_range && dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    // Resource collection
    if (best_idx != -1) {
        Resource& r = resources[best_idx];
        float grab_range = 0.02f + 0.02f * collect_strength;
        
        if (best_dist < grab_range*grab_range) {
            // Collect resource
            float bonus = 1.0f + 0.5f * collect_strength; // 50% bonus for collectors
            
            // Territory bonus: defenders nearby
            int defenders_nearby = 0;
            for (int i = 0; i < AGENTS; i++) {
                if (i == idx) continue;
                Agent& other = agents[i];
                if (other.arch == a.arch && other.role[3] > 0.3f) {
                    float dx = a.x - other.x;
                    float dy = a.y - other.y;
                    if (dx*dx + dy*dy < 0.04f) {
                        defenders_nearby++;
                    }
                }
            }
            float territory_bonus = 1.0f + 0.2f * defenders_nearby;
            
            float gained = r.value * bonus * territory_bonus;
            a.energy += gained;
            a.fitness += gained;
            
            // Pheromone deposit at resource location (NOVEL)
            add_pheromone(r.x, r.y, 1.0f + collect_strength);
            
            r.collected = 1;
            r.pheromone_strength = 10; // Mark for respawn
            
            // Store in memory
            a.memory_x = r.x;
            a.memory_y = r.y;
            a.memory_valid = 1;
        } else {
            // Move toward resource
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            float len = sqrtf(dx*dx + dy*dy) + 0.0001f;
            a.vx += dx / len * 0.001f * collect_strength;
            a.vy += dy / len * 0.001f * collect_strength;
        }
    }
    
    // Communication
    if (comm_strength > 0.3f && a.memory_valid) {
        // Broadcast memory to nearby agents of same archetype
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent& other = agents[i];
            if (other.arch == a.arch) {
                float dx = a.x - other.x;
                float dy = a.y - other.y;
                if (dx*dx + dy*dy < 0.0036f) { // 0.06 radius
                    other.memory_x = a.memory_x;
                    other.memory_y = a.memory_y;
                    other.memory_valid = 1;
                }
            }
        }
    }
    
    // Use memory if no immediate resource found
    if (best_idx == -1 && a.memory_valid) {
        float dx = a.memory_x - a.x;
        float dy = a.memory_y - a.y;
        float dist = dx*dx + dy*dy;
        
        if (dist > 0.0001f) {
            float len = sqrtf(dist);
            a.vx += dx / len * 0.0005f * explore_strength;
            a.vy += dy / len * 0.0005f * explore_strength;
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0 && lcgf(a.rng) < 0.1f) {
        // Defenders resist perturbation
        if (defend_strength < 0.3f) {
            a.energy *= 0.5f;
            a.vx += (lcgf(a.rng) - 0.5f) * 0.1f;
            a.vy += (lcgf(a.rng) - 0.5f) * 0.1f;
        }
    }
}

// Resource respawn
__global__ void respawn_resources(Resource* resources) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    Resource& r = resources[idx];
    if (r.collected) {
        r.pheromone_strength--;
        if (r.pheromone_strength <= 0) {
            r.x = lcgf(r.rng);
            r.y = lcgf(r.rng);
            r.value = 0.5f + lcgf(r.rng) * 0.5f;
            r.collected = 0;
            
            // Add initial pheromone at new resource location
            add_pheromone(r.x, r.y, 0.5f);
        }
    }
}

// Calculate statistics
__global__ void calculate_stats(Agent* agents, float* stats) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    
    // Contribution to fitness sum
    atomicAdd(&stats[0], a.fitness);
    
    // Contribution to specialization measure
    float max_role = 0.0f;
    for (int i = 0; i < 4;