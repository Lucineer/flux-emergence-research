// CUDA Simulation Experiment v70: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist efficiency
// Prediction: Pheromones will help uniform agents more than specialists (reducing specialist advantage)
// because uniform agents can follow trails without coordination costs
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Novel: Agents leave pheromone markers at collected resources that decay over time
//        All agents can detect nearby pheromones and move toward strongest concentration

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 64; // 64x64 grid for pheromone field
const float WORLD_SIZE = 1.0f;
const float CELL_SIZE = WORLD_SIZE / PHEROMONE_GRID;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Dominant archetype
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone structure for 2D grid
struct PheromoneGrid {
    float trail[PHEROMONE_GRID * PHEROMONE_GRID];
};

// Linear congruential generator
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int num_agents, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent* a = &agents[idx];
    a->rng = seed + idx * 137;
    
    // Random position
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    // Specialized group (first half) vs uniform control (second half)
    if (idx < num_agents / 2) {
        // Specialized: role[arch] = 0.7, others = 0.1
        a->arch = idx % 4;  // Even distribution of archetypes
        for (int i = 0; i < 4; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform: all roles = 0.25
        a->arch = -1;  // No dominant archetype
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources, int num_res, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_res) return;
    
    Resource* r = &resources[idx];
    unsigned int rng = seed + idx * 7919;
    
    // Uniform random distribution
    r->x = (lcg(&rng) / 4294967296.0f);
    r->y = (lcg(&rng) / 4294967296.0f);
    r->value = 0.5f + (lcg(&rng) / 4294967296.0f) * 0.5f;  // 0.5 to 1.0
    r->collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(PheromoneGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    grid->trail[idx] = 0.0f;
}

// Decay pheromones kernel
__global__ void decay_pheromones(PheromoneGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    // Exponential decay: 5% per tick
    grid->trail[idx] *= 0.95f;
}

// Deposit pheromone at resource location
__device__ void deposit_pheromone(PheromoneGrid* grid, float x, float y, float amount) {
    int gx = (int)(x / CELL_SIZE);
    int gy = (int)(y / CELL_SIZE);
    
    if (gx >= 0 && gx < PHEROMONE_GRID && gy >= 0 && gy < PHEROMONE_GRID) {
        int idx = gy * PHEROMONE_GRID + gx;
        atomicAdd(&grid->trail[idx], amount);
    }
}

// Sample pheromone at position
__device__ float sample_pheromone(PheromoneGrid* grid, float x, float y) {
    int gx = (int)(x / CELL_SIZE);
    int gy = (int)(y / CELL_SIZE);
    
    if (gx >= 0 && gx < PHEROMONE_GRID && gy >= 0 && gy < PHEROMONE_GRID) {
        return grid->trail[gy * PHEROMONE_GRID + gx];
    }
    return 0.0f;
}

// Get pheromone gradient direction
__device__ void get_pheromone_gradient(PheromoneGrid* grid, float x, float y, float* dx, float* dy) {
    float center = sample_pheromone(grid, x, y);
    float right = sample_pheromone(grid, x + CELL_SIZE, y);
    float left = sample_pheromone(grid, x - CELL_SIZE, y);
    float up = sample_pheromone(grid, x, y + CELL_SIZE);
    float down = sample_pheromone(grid, x, y - CELL_SIZE);
    
    *dx = (right - left) / (2.0f * CELL_SIZE);
    *dy = (up - down) / (2.0f * CELL_SIZE);
    
    // Normalize
    float len = sqrtf(*dx * *dx + *dy * *dy);
    if (len > 1e-6f) {
        *dx /= len;
        *dy /= len;
    }
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, PheromoneGrid* pheromones,
                     int num_agents, int num_res, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect role similarity > 0.9, apply drift
    float max_role = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < 4; i++) {
        if (a->role[i] > max_role) {
            max_role = a->role[i];
            max_idx = i;
        }
    }
    
    if (max_role > 0.9f) {
        // Apply random drift to non-dominant roles
        for (int i = 0; i < 4; i++) {
            if (i != max_idx) {
                float drift = (lcgf(&a->rng) - 0.5f) * 0.02f;
                a->role[i] += drift;
                a->role[i] = fmaxf(0.05f, fminf(0.95f, a->role[i]));
            }
        }
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) {
            a->role[i] /= sum;
        }
    }
    
    // Role-based behavior probabilities
    float explore_prob = a->role[0];
    float collect_prob = a->role[1];
    float comm_prob = a->role[2];
    float defend_prob = a->role[3];
    
    // Pheromone following (NEW MECHANISM)
    float ph_dx = 0.0f, ph_dy = 0.0f;
    get_pheromone_gradient(pheromones, a->x, a->y, &ph_dx, &ph_dy);
    
    // Blend pheromone gradient with random exploration
    float ph_strength = sample_pheromone(pheromones, a->x, a->y);
    float ph_weight = fminf(ph_strength * 2.0f, 0.5f); // Max 50% influence
    
    // Random movement with pheromone influence
    float rand_dx = lcgf(&a->rng) * 0.02f - 0.01f;
    float rand_dy = lcgf(&a->rng) * 0.02f - 0.01f;
    
    a->vx = rand_dx * (1.0f - ph_weight) + ph_dx * 0.01f * ph_weight;
    a->vy = rand_dy * (1.0f - ph_weight) + ph_dy * 0.01f * ph_weight;
    
    // Velocity limits
    float speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
    if (speed > 0.015f) {
        a->vx *= 0.015f / speed;
        a->vy *= 0.015f / speed;
    }
    
    // Update position with wrap-around
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0) a->x += 1.0f;
    if (a->x >= 1.0f) a->x -= 1.0f;
    if (a->y < 0) a->y += 1.0f;
    if (a->y >= 1.0f) a->y -= 1.0f;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    // Find nearest resource
    for (int i = 0; i < num_res; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx * dx + dy * dy);
        
        // Detection range based on explore role
        float detect_range = 0.03f + explore_prob * 0.04f;
        
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    // Collect resource if in range
    if (best_res != -1) {
        Resource* r = &resources[best_res];
        float grab_range = 0.02f + collect_prob * 0.02f;
        
        if (best_dist < grab_range) {
            // Collection bonus for collectors
            float bonus = 1.0f + collect_prob * 0.5f;
            float gained = r->value * bonus;
            
            a->energy += gained;
            a->fitness += gained;
            r->collected = 1;
            
            // DEPOSIT PHEROMONE AT COLLECTED RESOURCE (NEW MECHANISM)
            deposit_pheromone(pheromones, r->x, r->y, 1.0f);
        }
    }
    
    // Communication behavior
    if (lcgf(&a->rng) < comm_prob * 0.1f) {
        // Find nearest agent of same archetype
        float best_agent_dist = 0.2f;
        int best_agent = -1;
        
        for (int i = 0; i < num_agents; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            
            // Coupling: stronger for same archetype
            float coupling = (a->arch == other->arch) ? 0.02f : 0.002f;
            
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            if (dx > 0.5f) dx -= 1.0f;
            if (dx < -0.5f) dx += 1.0f;
            if (dy > 0.5f) dy -= 1.0f;
            if (dy < -0.5f) dy += 1.0f;
            
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist < 0.06f && dist < best_agent_dist) {
                best_agent_dist = dist;
                best_agent = i;
            }
        }
        
        // Share resource location if found one
        if (best_agent != -1 && best_res != -1) {
            Agent* other = &agents[best_agent];
            // Small energy transfer as incentive
            float transfer = 0.01f;
            if (a->energy > transfer) {
                a->energy -= transfer;
                other->energy += transfer;
            }
        }
    }
    
    // Defense behavior
    if (defend_prob > 0.3f) {
        // Territory boost: count nearby defenders of same archetype
        int nearby_defenders = 0;
        for (int i = 0; i < num_agents; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            if (other->role[3] > 0.3f && a->arch == other->arch) {
                float dx = other->x - a->x;
                float dy = other->y - a->y;
                if (dx > 0.5f) dx -= 1.0f;
                if (dx < -0.5f) dx += 1.0f;
                if (dy > 0.5f) dy -= 1.0f;
                if (dy < -0.5f) dy += 1.0f;
                
                float dist = sqrtf(dx * dx + dy * dy);
                if (dist < 0.1f) {
                    nearby_defenders++;
                }
            }
        }
        
        // Defense bonus: 20% per nearby defender
        float defense_bonus = 1.0f + nearby_defenders * 0.2f;
        a->energy *= defense_bonus;
        
        // Perturbation resistance
        if (tick_num % 100 == 0) { // Periodic perturbations
            // Defenders resist energy halving
            float resistance = defend_prob;
            a->energy *= (0.5f + 0.5f * resistance);
        }
    }
    
    // Energy limits
    if (a->energy > 2.0f) a->energy = 2.0f;
    if (a->energy < 0.01f) a->energy = 0.01f;
}

// Reset resources periodically
__global__ void reset_resources(Resource* resources, int num_res, unsigned int seed, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_res) return;
    
    Resource* r = &resources[idx];
    unsigned int rng = seed + idx * 7919 + tick_num;
    
    // Respawn every 50 ticks if collected
    if (r->collected && (tick_num % 50 == 0)) {
        r->x = (lcg(&rng) / 4294967296.0f);
        r->y = (lcg(&rng) / 4294967296.0f);
        r->value = 0.5f + (lcg(&rng) / 4294967296.0f) * 0.5f;
        r->collected = 0;
    }
}

int main