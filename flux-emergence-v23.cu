
/*
CUDA Simulation Experiment v23: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents by >1.61x (v8 baseline) due to improved resource location memory.
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence) included.
Novelty: Stigmergy - agents deposit pheromones when collecting resources, others sense them.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 64; // 64x64 grid for pheromone field
const float WORLD_SIZE = 1.0f;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_DEPOSIT = 10.0f;
const float SENSE_PHEROMONE_RANGE = 0.08f;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent structure
struct Agent {
    float x, y;           // position
    float vx, vy;         // velocity
    float energy;         // energy level
    float role[4];        // behavioral roles: explore, collect, communicate, defend
    float fitness;        // fitness score
    int arch;             // archetype (0-3)
    unsigned int rng;     // random state
};

// Resource structure
struct Resource {
    float x, y;           // position
    float value;          // resource value
    int collected;        // collected flag
};

// Pheromone structure
struct PheromoneGrid {
    float trail[PHEROMONE_GRID][PHEROMONE_GRID];
};

// Linear congruential generator (device/host)
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent *agents, PheromoneGrid *pheromone, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    unsigned int rng = seed + idx * 17;
    
    agents[idx].x = lcgf(rng);
    agents[idx].y = lcgf(rng);
    agents[idx].vx = lcgf(rng) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(rng) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].rng = rng;
    
    // Specialized agents (first half) vs uniform control (second half)
    if (idx < AGENTS/2) {
        // Specialized: role[arch] = 0.7, others 0.1
        agents[idx].arch = idx % 4;
        for (int i = 0; i < 4; i++) agents[idx].role[i] = 0.1f;
        agents[idx].role[agents[idx].arch] = 0.7f;
    } else {
        // Uniform: all roles = 0.25
        agents[idx].arch = idx % 4;
        for (int i = 0; i < 4; i++) agents[idx].role[i] = 0.25f;
    }
    
    // Initialize pheromone grid to zero
    if (idx == 0) {
        for (int i = 0; i < PHEROMONE_GRID; i++)
            for (int j = 0; j < PHEROMONE_GRID; j++)
                pheromone->trail[i][j] = 0.0f;
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource *resources, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = seed + idx * 29;
    resources[idx].x = lcgf(rng);
    resources[idx].y = lcgf(rng);
    resources[idx].value = 0.8f + lcgf(rng) * 0.4f; // 0.8-1.2
    resources[idx].collected = 0;
}

// Decay pheromones kernel
__global__ void decay_pheromones(PheromoneGrid *pheromone) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < PHEROMONE_GRID && j < PHEROMONE_GRID) {
        pheromone->trail[i][j] *= PHEROMONE_DECAY;
    }
}

// Get pheromone value at position
__device__ float get_pheromone(float x, float y, PheromoneGrid *pheromone) {
    int gx = (int)(x * PHEROMONE_GRID) % PHEROMONE_GRID;
    int gy = (int)(y * PHEROMONE_GRID) % PHEROMONE_GRID;
    if (gx < 0) gx += PHEROMONE_GRID;
    if (gy < 0) gy += PHEROMONE_GRID;
    return pheromone->trail[gx][gy];
}

// Add pheromone at position
__device__ void add_pheromone(float x, float y, float amount, PheromoneGrid *pheromone) {
    int gx = (int)(x * PHEROMONE_GRID) % PHEROMONE_GRID;
    int gy = (int)(y * PHEROMONE_GRID) % PHEROMONE_GRID;
    if (gx < 0) gx += PHEROMONE_GRID;
    if (gy < 0) gy += PHEROMONE_GRID;
    atomicAdd(&pheromone->trail[gx][gy], amount);
}

// Main simulation tick kernel
__global__ void tick(Agent *agents, Resource *resources, PheromoneGrid *pheromone, 
                     int tick_num, float *specialist_energy, float *uniform_energy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    unsigned int &rng = a.rng;
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0) {
        float resist = 1.0f - a.role[3] * 0.5f; // defenders resist
        a.energy *= (0.5f + 0.5f * resist);
    }
    
    // Anti-convergence: check similarity with random other agent
    int other = lcg(rng) % AGENTS;
    if (other != idx) {
        float similarity = 0.0f;
        for (int i = 0; i < 4; i++) similarity += fabsf(a.role[i] - agents[other].role[i]);
        similarity = 1.0f - similarity / 4.0f;
        
        if (similarity > 0.9f) {
            // Find non-dominant role
            int non_dom = 0;
            for (int i = 1; i < 4; i++) if (a.role[i] < a.role[non_dom]) non_dom = i;
            a.role[non_dom] += (lcgf(rng) * 0.02f - 0.01f);
            // Renormalize
            float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
            for (int i = 0; i < 4; i++) a.role[i] /= sum;
        }
    }
    
    // Sense pheromones (NOVEL MECHANISM)
    float pheromone_strength = 0.0f;
    float pheromone_dir_x = 0.0f, pheromone_dir_y = 0.0f;
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            float sx = a.x + dx * SENSE_PHEROMONE_RANGE;
            float sy = a.y + dy * SENSE_PHEROMONE_RANGE;
            float p = get_pheromone(sx, sy, pheromone);
            pheromone_strength += p;
            pheromone_dir_x += p * dx;
            pheromone_dir_y += p * dy;
        }
    }
    
    // Movement influenced by pheromones and roles
    float explore_weight = a.role[0];
    float pheromone_weight = 0.3f * a.role[1]; // collectors follow pheromones more
    
    // Random exploration
    a.vx += (lcgf(rng) - 0.5f) * 0.01f * explore_weight;
    a.vy += (lcgf(rng) - 0.5f) * 0.01f * explore_weight;
    
    // Pheromone following (NOVEL)
    if (pheromone_strength > 0.1f) {
        a.vx += pheromone_dir_x * 0.005f * pheromone_weight;
        a.vy += pheromone_dir_y * 0.005f * pheromone_weight;
    }
    
    // Velocity damping and bounds
    a.vx *= 0.95f;
    a.vy *= 0.95f;
    a.x += a.vx;
    a.y += a.vy;
    
    // World wrap
    if (a.x < 0) a.x += 1.0f;
    if (a.x >= 1.0f) a.x -= 1.0f;
    if (a.y < 0) a.y += 1.0f;
    if (a.y >= 1.0f) a.y -= 1.0f;
    
    // Resource interaction
    float detect_range = 0.03f + a.role[0] * 0.04f; // explorers detect better
    float grab_range = 0.02f + a.role[1] * 0.02f;   // collectors grab better
    
    float best_dist = 1e6;
    int best_res = -1;
    
    // Find nearest resource
    for (int r = 0; r < RESOURCES; r++) {
        if (resources[r].collected) continue;
        
        float dx = a.x - resources[r].x;
        float dy = a.y - resources[r].y;
        // Wrap distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < best_dist) {
            best_dist = dist;
            best_res = r;
        }
    }
    
    if (best_res != -1) {
        Resource &res = resources[best_res];
        
        // Detection
        if (best_dist < detect_range) {
            // Communication
            if (a.role[2] > 0.3f) {
                // Broadcast to nearby agents
                for (int n = 0; n < AGENTS; n++) {
                    if (n == idx) continue;
                    Agent &other = agents[n];
                    float dx = a.x - other.x;
                    float dy = a.y - other.y;
                    if (dx > 0.5f) dx -= 1.0f;
                    if (dx < -0.5f) dx += 1.0f;
                    if (dy > 0.5f) dy -= 1.0f;
                    if (dy < -0.5f) dy += 1.0f;
                    
                    if (sqrtf(dx*dx + dy*dy) < 0.06f) {
                        // Attract toward resource
                        float dir_x = res.x - other.x;
                        float dir_y = res.y - other.y;
                        if (dir_x > 0.5f) dir_x -= 1.0f;
                        if (dir_x < -0.5f) dir_x += 1.0f;
                        if (dir_y > 0.5f) dir_y -= 1.0f;
                        if (dir_y < -0.5f) dir_y += 1.0f;
                        
                        float len = sqrtf(dir_x*dir_x + dir_y*dir_y);
                        if (len > 0.001f) {
                            other.vx += dir_x / len * 0.01f * a.role[2];
                            other.vy += dir_y / len * 0.01f * a.role[2];
                        }
                    }
                }
            }
            
            // Collection
            if (best_dist < grab_range) {
                float base_gain = res.value;
                float collector_bonus = 1.0f + a.role[1] * 0.5f;
                
                // Territory bonus from nearby defenders
                float territory_bonus = 1.0f;
                for (int n = 0; n < AGENTS; n++) {
                    if (n == idx) continue;
                    Agent &other = agents[n];
                    if (other.arch == ARCH_DEFENDER) {
                        float dx = a.x - other.x;
                        float dy = a.y - other.y;
                        if (dx > 0.5f) dx -= 1.0f;
                        if (dx < -0.5f) dx += 1.0f;
                        if (dy > 0.5f) dy -= 1.0f;
                        if (dy < -0.5f) dy += 1.0f;
                        
                        if (sqrtf(dx*dx + dy*dy) < 0.05f) {
                            territory_bonus += 0.2f;
                        }
                    }
                }
                
                float gain = base_gain * collector_bonus * territory_bonus;
                a.energy += gain;
                a.fitness += gain;
                res.collected = 1;
                
                // DEPOSIT PHEROMONE AT COLLECTION SITE (NOVEL)
                add_pheromone(res.x, res.y, PHEROMONE_DEPOSIT, pheromone);
            }
        }
    }
    
    // Energy coupling with similar agents
    for (int n = 0; n < AGENTS; n++) {
        if (n == idx) continue;
        Agent &other = agents[n];
        float dx = a.x - other.x;
        float dy = a.y - other.y;
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 0.02f) {
            float coupling = (a.arch == other.arch) ? 0.02f : 0.002f;
            float transfer = (other.energy - a.energy) * coupling;
            a.energy += transfer;
        }
    }
    
    // Track energy sums for comparison
    if (idx < AGENTS/2) {
        atomicAdd(specialist_energy, a.energy);
    } else {
        atomicAdd(uniform_energy, a.energy);
    }
}

// Resource respawn kernel
__global__ void respawn_resources(Resource *resources, int tick_num, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    // Respawn every 50 ticks
    if (tick_num % 50 == 0) {
        unsigned int rng = seed + idx * 37 + tick_num;
        resources[idx].collected = 0;
        // Occasionally move resources
        if (lcgf(rng) < 0.3f) {
            resources[idx].x = lcgf(rng);
            resources[idx].y = lcgf(rng);
        }
    }
}

int main() {
    // Allocate memory
    Agent *d_agents;
    Resource *d_resources;
    PheromoneGrid *d_pheromone;
    float *d_specialist_energy, *d_uniform_energy;
    float h_specialist_energy, h_uniform_energy;
    
    cudaMalloc(&d_agents, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    cudaMalloc(&d_pherom