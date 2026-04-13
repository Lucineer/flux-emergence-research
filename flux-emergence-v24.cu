// CUDA Simulation Experiment v24: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone markers at resource locations that decay over time
// Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents
// Novel mechanism: Stigmergy (indirect communication through environment modification)
// Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence)

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const float WORLD_SIZE = 1.0f;
const float MIN_DIST = 0.0001f;

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Archetype (0-3)
    unsigned int rng;     // Random number state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone structure (NEW for v24)
struct Pheromone {
    float x, y;           // Location
    float strength;       // Current strength
    int arch;             // Archetype that left it
    int age;              // Age in ticks
};

// RNG functions
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
    a->x = lcgf(&a->rng) * WORLD_SIZE;
    a->y = lcgf(&a->rng) * WORLD_SIZE;
    a->vx = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->vy = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    a->arch = idx % ARCHETYPES;
    
    // Specialized agents (first half): strong in their archetype's role
    if (idx < num_agents / 2) {
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
        a->role[a->arch] = 0.7f;  // Specialized in own archetype
    }
    // Uniform control agents (second half): all roles equal
    else {
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources, int num_res, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_res) return;
    
    unsigned int rng = seed + idx * 7919;
    resources[idx].x = lcgf(&rng) * WORLD_SIZE;
    resources[idx].y = lcgf(&rng) * WORLD_SIZE;
    resources[idx].value = 0.5f + lcgf(&rng) * 0.5f;
    resources[idx].collected = 0;
}

// Initialize pheromones kernel (NEW for v24)
__global__ void init_pheromones(Pheromone* pheromones, int max_pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_pheromones) return;
    
    pheromones[idx].strength = 0.0f;
    pheromones[idx].age = 0;
}

// Update pheromones kernel (NEW for v24)
__global__ void update_pheromones(Pheromone* pheromones, int* pheromone_count, int max_pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_pheromones) return;
    
    if (pheromones[idx].strength > 0.0f) {
        // Decay pheromone strength
        pheromones[idx].strength *= 0.95f;
        pheromones[idx].age++;
        
        // Remove old pheromones
        if (pheromones[idx].strength < 0.01f || pheromones[idx].age > 100) {
            pheromones[idx].strength = 0.0f;
            atomicSub(pheromone_count, 1);
        }
    }
}

// Add pheromone at location (NEW for v24)
__device__ void add_pheromone(Pheromone* pheromones, int* pheromone_count, int max_pheromones,
                             float x, float y, int arch, unsigned int* rng) {
    // Find empty slot
    for (int i = 0; i < max_pheromones; i++) {
        int idx = (i + arch * 37) % max_pheromones;
        if (pheromones[idx].strength == 0.0f) {
            pheromones[idx].x = x;
            pheromones[idx].y = y;
            pheromones[idx].strength = 1.0f;
            pheromones[idx].arch = arch;
            pheromones[idx].age = 0;
            atomicAdd(pheromone_count, 1);
            break;
        }
    }
}

// Main simulation kernel
__global__ void tick_kernel(Agent* agents, int num_agents, Resource* resources, int num_res,
                           Pheromone* pheromones, int* pheromone_count, int max_pheromones,
                           int tick, float* specialist_energy, float* uniform_energy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect role similarity with neighbors
    int similar_count = 0;
    for (int i = 0; i < num_agents; i += num_agents / 16) {  // Sample 16 agents
        if (i == idx) continue;
        Agent* other = &agents[i];
        float dx = a->x - other->x;
        float dy = a->y - other->y;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 0.1f) {
            float role_diff = 0.0f;
            for (int r = 0; r < 4; r++) {
                role_diff += fabsf(a->role[r] - other->role[r]);
            }
            if (role_diff < 0.4f) similar_count++;
        }
    }
    
    // Apply anti-convergence drift if too similar
    if (similar_count > 5) {
        int drift_role = (int)(lcgf(&a->rng) * 4);
        if (drift_role != a->arch) {  // Don't drift dominant role
            a->role[drift_role] += (lcgf(&a->rng) - 0.5f) * 0.02f;
            a->role[drift_role] = fmaxf(0.0f, fminf(1.0f, a->role[drift_role]));
        }
    }
    
    // Normalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) a->role[i] /= sum;
    
    // Behavioral roles
    float explore_strength = a->role[0];
    float collect_strength = a->role[1];
    float comm_strength = a->role[2];
    float defend_strength = a->role[3];
    
    // Explore behavior: move randomly but follow pheromones of same archetype (NEW for v24)
    float explore_vx = (lcgf(&a->rng) - 0.5f) * 0.04f;
    float explore_vy = (lcgf(&a->rng) - 0.5f) * 0.04f;
    
    // Pheromone following (NEW for v24)
    float best_pheromone_strength = 0.0f;
    float best_pheromone_dx = 0.0f;
    float best_pheromone_dy = 0.0f;
    
    for (int i = 0; i < max_pheromones; i++) {
        if (pheromones[i].strength > best_pheromone_strength && pheromones[i].arch == a->arch) {
            float dx = pheromones[i].x - a->x;
            float dy = pheromones[i].y - a->y;
            if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
            if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
            if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
            if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < 0.3f && dist > MIN_DIST) {
                best_pheromone_strength = pheromones[i].strength;
                best_pheromone_dx = dx / dist;
                best_pheromone_dy = dy / dist;
            }
        }
    }
    
    // Blend random exploration with pheromone following
    explore_vx = explore_vx * (1.0f - explore_strength) + best_pheromone_dx * 0.02f * explore_strength;
    explore_vy = explore_vy * (1.0f - explore_strength) + best_pheromone_dy * 0.02f * explore_strength;
    
    // Collect behavior: move toward nearest resource
    float collect_vx = 0.0f, collect_vy = 0.0f;
    float nearest_dist = 1.0f;
    float nearest_dx = 0.0f, nearest_dy = 0.0f;
    
    for (int i = 0; i < num_res; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < nearest_dist && dist < 0.07f * collect_strength) {
            nearest_dist = dist;
            nearest_dx = dx;
            nearest_dy = dy;
        }
    }
    
    if (nearest_dist < 1.0f && nearest_dist > MIN_DIST) {
        collect_vx = nearest_dx / nearest_dist * 0.03f;
        collect_vy = nearest_dy / nearest_dist * 0.03f;
    }
    
    // Communicate behavior: broadcast resource locations to nearby agents
    if (comm_strength > 0.1f) {
        for (int i = 0; i < num_agents; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            if (other->arch != a->arch) continue;
            
            float dx = a->x - other->x;
            float dy = a->y - other->y;
            if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
            if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
            if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
            if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < 0.06f * comm_strength) {
                // Share nearest resource location
                if (nearest_dist < 0.5f) {
                    // Influence other agent's velocity toward resource
                    other->vx += nearest_dx / nearest_dist * 0.01f * comm_strength;
                    other->vy += nearest_dy / nearest_dist * 0.01f * comm_strength;
                }
            }
        }
    }
    
    // Defend behavior: territory and perturbation resistance
    float defend_vx = 0.0f, defend_vy = 0.0f;
    int defender_count = 0;
    
    for (int i = 0; i < num_agents; i++) {
        if (i == idx) continue;
        Agent* other = &agents[i];
        if (other->arch != a->arch) continue;
        
        float dx = a->x - other->x;
        float dy = a->y - other->y;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 0.05f && other->role[3] > 0.3f) {
            defender_count++;
            // Move away from same-arch defenders to spread out
            defend_vx -= dx / dist * 0.01f;
            defend_vy -= dy / dist * 0.01f;
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick % 50 == 0) {
        float resistance = 1.0f - defend_strength * 0.5f;
        a->energy *= 0.5f + 0.5f * resistance;
        a->vx += (lcgf(&a->rng) - 0.5f) * 0.1f * (1.0f - resistance);
        a->vy += (lcgf(&a->rng) - 0.5f) * 0.1f * (1.0f - resistance);
    }
    
    // Combine behaviors
    a->vx = a->vx * 0.7f + explore_vx * explore_strength + collect_vx * collect_strength + 
            defend_vx * defend_strength;
    a->vy = a->vy * 0.7f + explore_vy * explore_strength + collect_vy * collect_strength + 
            defend_vy * defend_strength;
    
    // Limit velocity
    float speed = sqrtf(a->vx*a->vx + a->vy*a->vy);
    if (speed > 0.03f) {
        a->vx *= 0.03f / speed;
        a->vy *= 0.03f / speed;
    }
    
    // Update position with wrap-around
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0) a->x += WORLD_SIZE;
    if (a->x >= WORLD_SIZE) a->x -= WORLD_SIZE;
    if (a->y < 0) a->y += WORLD_SIZE;
    if (a->y >= WORLD_SIZE) a->y -= WORLD_SIZE;
    
    // Resource collection
    for (int i = 0; i < num_res; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        float grab_range = 0.02f + collect