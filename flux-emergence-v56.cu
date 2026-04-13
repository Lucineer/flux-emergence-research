/*
CUDA Simulation Experiment v56: Stigmergy with Pheromone Trails
Testing: Whether pheromone trails left at resource locations improve specialist efficiency
Prediction: Pheromones will help uniform agents more than specialists (reducing specialist advantage)
    because specialists already have optimized detection, while uniform agents benefit from shared info.
Baseline: v8 mechanisms (scarcity, territory, communication) + anti-convergence
Novelty: Agents leave pheromone markers at collected resource locations that decay over time
    Other agents can detect pheromone concentration to find resources
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants for sm_87 (Jetson Orin)
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Pheromone grid constants
const int PHEROMONE_GRID_SIZE = 256;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_STRENGTH = 10.0f;
const float PHEROMONE_DETECTION_RANGE = 0.08f;

// Agent archetypes
enum { ARCH_UNIFORM = 0, ARCH_SPECIALIST = 1 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: [explore, collect, communicate, defend]
    float fitness;        // Fitness score
    int arch;             // Archetype
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone grid structure
struct PheromoneGrid {
    float concentration[PHEROMONE_GRID_SIZE][PHEROMONE_GRID_SIZE];
};

// Linear Congruential Generator (LCG)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

// LCG float in [0,1)
__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize pheromone grid
__global__ void initPheromone(PheromoneGrid* grid) {
    for (int i = threadIdx.x; i < PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE; i += blockDim.x) {
        int x = i % PHEROMONE_GRID_SIZE;
        int y = i / PHEROMONE_GRID_SIZE;
        grid->concentration[x][y] = 0.0f;
    }
}

// Decay pheromones
__global__ void decayPheromones(PheromoneGrid* grid) {
    for (int i = threadIdx.x; i < PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE; i += blockDim.x) {
        int x = i % PHEROMONE_GRID_SIZE;
        int y = i / PHEROMONE_GRID_SIZE;
        grid->concentration[x][y] *= PHEROMONE_DECAY;
    }
}

// Initialize agents and resources
__global__ void init(Agent* agents, Resource* resources, PheromoneGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < AGENTS) {
        // Initialize agent
        agents[idx].x = lcgf(&agents[idx].rng);
        agents[idx].y = lcgf(&agents[idx].rng);
        agents[idx].vx = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
        agents[idx].vy = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
        agents[idx].energy = 1.0f;
        agents[idx].fitness = 0.0f;
        agents[idx].rng = idx * 123456789u + 987654321u;
        
        // Assign archetype (half uniform, half specialist)
        agents[idx].arch = (idx < AGENTS/2) ? ARCH_UNIFORM : ARCH_SPECIALIST;
        
        // Set roles based on archetype
        if (agents[idx].arch == ARCH_UNIFORM) {
            // Uniform: all roles equal
            for (int i = 0; i < 4; i++) {
                agents[idx].role[i] = 0.25f;
            }
        } else {
            // Specialist: one dominant role per agent (randomly assigned)
            int dominant = idx % 4;
            for (int i = 0; i < 4; i++) {
                agents[idx].role[i] = (i == dominant) ? 0.7f : 0.1f;
            }
        }
    }
    
    if (idx < RESOURCES) {
        // Initialize resources (scattered uniformly)
        unsigned int rng = idx * 135791113u + 171923293u;
        resources[idx].x = lcgf(&rng);
        resources[idx].y = lcgf(&rng);
        resources[idx].value = 0.5f + lcgf(&rng) * 0.5f;  // 0.5-1.0
        resources[idx].collected = 0;
    }
    
    if (idx == 0) {
        // Initialize pheromone grid
        for (int i = 0; i < PHEROMONE_GRID_SIZE; i++) {
            for (int j = 0; j < PHEROMONE_GRID_SIZE; j++) {
                grid->concentration[i][j] = 0.0f;
            }
        }
    }
}

// Get pheromone concentration at position
__device__ float getPheromone(PheromoneGrid* grid, float x, float y) {
    int gx = min(max((int)(x * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int gy = min(max((int)(y * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    return grid->concentration[gx][gy];
}

// Add pheromone at position
__device__ void addPheromone(PheromoneGrid* grid, float x, float y, float amount) {
    int gx = min(max((int)(x * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int gy = min(max((int)(y * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    atomicAdd(&grid->concentration[gx][gy], amount);
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, PheromoneGrid* grid, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = (int)(lcgf(&a->rng) * AGENTS);
    if (other_idx >= AGENTS) other_idx = AGENTS - 1;
    Agent* other = &agents[other_idx];
    
    if (other->arch == a->arch) {
        // Calculate role similarity
        float similarity = 0.0f;
        for (int i = 0; i < 4; i++) {
            similarity += 1.0f - fabsf(a->role[i] - other->role[i]);
        }
        similarity /= 4.0f;
        
        // If too similar, apply random drift to non-dominant roles
        if (similarity > 0.9f) {
            int dominant = 0;
            for (int i = 1; i < 4; i++) {
                if (a->role[i] > a->role[dominant]) dominant = i;
            }
            
            for (int i = 0; i < 4; i++) {
                if (i != dominant) {
                    a->role[i] += (lcgf(&a->rng) * 0.02f - 0.01f);
                    a->role[i] = max(0.0f, min(1.0f, a->role[i]));
                }
            }
            
            // Renormalize
            float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
            for (int i = 0; i < 4; i++) {
                a->role[i] /= sum;
            }
        }
    }
    
    // Coupling: adjust roles toward same archetype, away from different
    float coupling = (other->arch == a->arch) ? 0.02f : 0.002f;
    for (int i = 0; i < 4; i++) {
        a->role[i] += (other->role[i] - a->role[i]) * coupling;
    }
    
    // Renormalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) {
        a->role[i] /= sum;
    }
    
    // Movement based on roles and pheromones
    float explore_strength = a->role[0];
    float collect_strength = a->role[1];
    float comm_strength = a->role[2];
    float defend_strength = a->role[3];
    
    // Pheromone influence: move toward high pheromone concentrations
    float pheromone_influence = 0.0f;
    
    // Sample pheromone in surrounding area
    float best_pheromone = 0.0f;
    float best_dx = 0.0f, best_dy = 0.0f;
    
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            float sample_x = a->x + i * PHEROMONE_DETECTION_RANGE;
            float sample_y = a->y + j * PHEROMONE_DETECTION_RANGE;
            
            if (sample_x >= 0.0f && sample_x <= 1.0f && 
                sample_y >= 0.0f && sample_y <= 1.0f) {
                float p = getPheromone(grid, sample_x, sample_y);
                if (p > best_pheromone) {
                    best_pheromone = p;
                    best_dx = i * PHEROMONE_DETECTION_RANGE;
                    best_dy = j * PHEROMONE_DETECTION_RANGE;
                }
            }
        }
    }
    
    // Apply pheromone influence (stronger for explorers)
    if (best_pheromone > 0.1f) {
        pheromone_influence = explore_strength * 0.5f;
        a->vx += best_dx * pheromone_influence;
        a->vy += best_dy * pheromone_influence;
    }
    
    // Random exploration
    a->vx += (lcgf(&a->rng) * 0.02f - 0.01f) * explore_strength;
    a->vy += (lcgf(&a->rng) * 0.02f - 0.01f) * explore_strength;
    
    // Velocity damping and position update
    a->vx *= 0.95f;
    a->vy *= 0.95f;
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary wrap
    if (a->x < 0.0f) a->x = 1.0f + a->x;
    if (a->x > 1.0f) a->x = a->x - 1.0f;
    if (a->y < 0.0f) a->y = 1.0f + a->y;
    if (a->y > 1.0f) a->y = a->y - 1.0f;
    
    // Resource interaction
    float detection_range = 0.03f + explore_strength * 0.04f;  // 0.03-0.07
    float grab_range = 0.02f + collect_strength * 0.02f;       // 0.02-0.04
    
    // Find nearest resource
    int nearest_res = -1;
    float nearest_dist = 1e6f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        dx = fminf(fabsf(dx), fminf(fabsf(dx + 1.0f), fabsf(dx - 1.0f)));
        dy = fminf(fabsf(dy), fminf(fabsf(dy + 1.0f), fabsf(dy - 1.0f)));
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detection_range && dist < nearest_dist) {
            nearest_dist = dist;
            nearest_res = i;
        }
    }
    
    // Collect resource if in range
    if (nearest_res != -1 && nearest_dist < grab_range) {
        Resource* r = &resources[nearest_res];
        
        // Collection bonus for collectors
        float bonus = 1.0f + collect_strength * 0.5f;  // Up to 50% bonus
        
        // Territory bonus for defenders
        int nearby_defenders = 0;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            if (other->arch != a->arch) continue;
            
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            dx = fminf(fabsf(dx), fminf(fabsf(dx + 1.0f), fabsf(dx - 1.0f)));
            dy = fminf(fabsf(dy), fminf(fabsf(dy + 1.0f), fabsf(dy - 1.0f)));
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < 0.1f && other->role[3] > 0.3f) {
                nearby_defenders++;
            }
        }
        
        float territory_bonus = 1.0f + nearby_defenders * 0.2f;  // 20% per defender
        
        // Calculate final energy gain
        float energy_gain = r->value * bonus * territory_bonus;
        a->energy += energy_gain;
        a->fitness += energy_gain;
        
        // Mark resource as collected
        r->collected = 1;
        
        // Leave pheromone at collected resource location
        addPheromone(grid, r->x, r->y, PHEROMONE_STRENGTH * collect_strength);
    }
    
    // Communication (broadcast nearest resource location)
    if (comm_strength > 0.3f && nearest_res != -1) {
        float comm_range = 0.06f;
        Resource* r = &resources[nearest_res];
        
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            if (other->arch != a->arch) continue;
            
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            dx = fminf(fabsf(dx), fminf(fabsf(dx + 1.0f), fabsf(dx - 1.0f)));
            dy = fminf(fabsf(dy), fminf(fabsf(dy + 1.0f), fabsf(dy - 1.0f)));
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < comm_range) {
                // Influence other agent toward resource
                float influence = comm_strength * 0.1f;
                float res_dx = r->x - other->x;
                float res_dy = r->y - other->y;
                
                // Wrap-around direction
                if (fabsf(res_dx) > 0.5f) res_dx = (res_dx > 0) ? res_dx - 1.0f : res_dx + 1.0f;
                if (fabsf(res_dy) > 0.5f) res_dy = (res_dy > 0) ? res_dy - 1.0f : res_dy + 1.0f;
                
                float len = sqrtf(res_dx*res_dx + res_dy*res_dy);
                if (len > 