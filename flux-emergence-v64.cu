// CUDA Simulation Experiment v64: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone trails at resource locations that decay over time
// Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents
// Expected: Specialists should show >1.61x advantage due to amplified communication via environmental markers

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;
const int PHEROMONE_GRID_SIZE = 256; // 256x256 grid for pheromone field
const float WORLD_SIZE = 1.0f;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3, ARCH_COUNT = 4 };

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
    int collected;        // Collection status
};

// Pheromone structure for stigmergy
struct Pheromone {
    float strength[ARCH_COUNT]; // Pheromone strength per archetype
    float decay;                // Decay rate
};

// LCG RNG functions
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
    a->x = lcgf(&a->rng) * WORLD_SIZE;
    a->y = lcgf(&a->rng) * WORLD_SIZE;
    
    // Random velocity
    float angle = lcgf(&a->rng) * 2.0f * M_PI;
    float speed = 0.001f + lcgf(&a->rng) * 0.002f;
    a->vx = cosf(angle) * speed;
    a->vy = sinf(angle) * speed;
    
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    // First half: specialized agents (role[arch]=0.7, others=0.1)
    // Second half: uniform control (all roles=0.25)
    if (idx < num_agents / 2) {
        a->arch = idx % ARCH_COUNT; // Assign archetype cyclically
        for (int i = 0; i < 4; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        a->arch = ARCH_EXPLORER; // All uniform agents are explorers (but roles are uniform)
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
    
    // Random position
    r->x = (lcg(&rng) / 4294967296.0f) * WORLD_SIZE;
    r->y = (lcg(&rng) / 4294967296.0f) * WORLD_SIZE;
    
    // Base value with some variation
    r->value = 0.8f + (lcg(&rng) / 4294967296.0f) * 0.4f;
    r->collected = 0;
}

// Initialize pheromone grid kernel
__global__ void init_pheromones(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    Pheromone* p = &grid[idx];
    for (int i = 0; i < ARCH_COUNT; i++) {
        p->strength[i] = 0.0f;
    }
    p->decay = 0.95f; // 5% decay per tick
}

// Decay pheromones kernel
__global__ void decay_pheromones(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    Pheromone* p = &grid[idx];
    for (int i = 0; i < ARCH_COUNT; i++) {
        p->strength[i] *= p->decay;
    }
}

// Deposit pheromone at resource location
__device__ void deposit_pheromone(Pheromone* grid, int grid_size, float x, float y, int arch, float amount) {
    int gx = (int)(x * grid_size);
    int gy = (int)(y * grid_size);
    gx = max(0, min(grid_size - 1, gx));
    gy = max(0, min(grid_size - 1, gy));
    
    int idx = gy * grid_size + gx;
    atomicAdd(&grid[idx].strength[arch], amount);
}

// Sample pheromone at position
__device__ float sample_pheromone(Pheromone* grid, int grid_size, float x, float y, int arch) {
    int gx = (int)(x * grid_size);
    int gy = (int)(y * grid_size);
    gx = max(0, min(grid_size - 1, gx));
    gy = max(0, min(grid_size - 1, gy));
    
    int idx = gy * grid_size + gx;
    return grid[idx].strength[arch];
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, int num_agents, Resource* resources, int num_res, 
                     Pheromone* pheromones, int grid_size, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Apply anti-convergence drift
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += a->role[i] * a->role[i];
    }
    similarity = sqrtf(similarity);
    
    if (similarity > 0.9f) {
        // Find non-dominant role
        int non_dominant = 0;
        for (int i = 1; i < 4; i++) {
            if (a->role[i] < a->role[non_dominant]) {
                non_dominant = i;
            }
        }
        // Apply random drift
        a->role[non_dominant] += (lcgf(&a->rng) - 0.5f) * 0.01f;
        
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) {
            a->role[i] /= sum;
        }
    }
    
    // Sample pheromone field (stigmergy mechanism)
    float arch_pheromone = sample_pheromone(pheromones, grid_size, a->x, a->y, a->arch);
    float pheromone_influence = arch_pheromone * 0.5f; // Scale influence
    
    // Movement with pheromone influence
    float explore_strength = a->role[0] * (0.03f + 0.04f * lcgf(&a->rng) + pheromone_influence);
    
    // Random walk with occasional pheromone-following
    if (lcgf(&a->rng) < 0.1f && arch_pheromone > 0.1f) {
        // Move toward higher pheromone concentration
        float dx = 0.0f, dy = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                float nx = a->x + i * 0.01f;
                float ny = a->y + j * 0.01f;
                if (nx < 0 || nx >= WORLD_SIZE || ny < 0 || ny >= WORLD_SIZE) continue;
                
                float p = sample_pheromone(pheromones, grid_size, nx, ny, a->arch);
                if (p > arch_pheromone) {
                    dx += i * (p - arch_pheromone);
                    dy += j * (p - arch_pheromone);
                }
            }
        }
        float len = sqrtf(dx*dx + dy*dy);
        if (len > 0) {
            a->vx = dx / len * explore_strength;
            a->vy = dy / len * explore_strength;
        }
    } else {
        // Normal random movement
        a->vx += (lcgf(&a->rng) - 0.5f) * 0.001f;
        a->vy += (lcgf(&a->rng) - 0.5f) * 0.001f;
    }
    
    // Velocity limiting
    float speed = sqrtf(a->vx*a->vx + a->vy*a->vy);
    if (speed > 0.005f) {
        a->vx *= 0.005f / speed;
        a->vy *= 0.005f / speed;
    }
    
    // Update position with wrap-around
    a->x += a->vx;
    a->y += a->vy;
    
    if (a->x < 0) a->x += WORLD_SIZE;
    if (a->x >= WORLD_SIZE) a->x -= WORLD_SIZE;
    if (a->y < 0) a->y += WORLD_SIZE;
    if (a->y >= WORLD_SIZE) a->y -= WORLD_SIZE;
    
    // Resource interaction
    float collect_strength = a->role[1] * (0.02f + 0.02f * lcgf(&a->rng));
    float comm_strength = a->role[2] * 0.06f;
    float defend_strength = a->role[3];
    
    // Find nearest resource
    int nearest_res = -1;
    float min_dist = 1.0f;
    
    for (int i = 0; i < num_res; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_res = i;
        }
    }
    
    if (nearest_res >= 0) {
        Resource* r = &resources[nearest_res];
        
        // Explore role: detect resources
        if (min_dist < explore_strength) {
            // Deposit pheromone at resource location (stigmergy)
            deposit_pheromone(pheromones, grid_size, r->x, r->y, a->arch, 1.0f);
        }
        
        // Collect role: grab resources
        if (min_dist < collect_strength) {
            // Collection bonus for specialists
            float bonus = (a->role[1] > 0.6f) ? 1.5f : 1.0f;
            float gain = r->value * bonus;
            
            // Territory bonus for defenders
            int nearby_defenders = 0;
            for (int j = 0; j < num_agents; j += num_agents/16) { // Sample other agents
                if (j == idx) continue;
                Agent* other = &agents[j];
                float dx = other->x - a->x;
                float dy = other->y - a->y;
                if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
                if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
                if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
                if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
                
                float dist = sqrtf(dx*dx + dy*dy);
                if (dist < 0.1f && other->arch == a->arch && other->role[3] > 0.6f) {
                    nearby_defenders++;
                }
            }
            gain *= (1.0f + 0.2f * nearby_defenders);
            
            a->energy += gain;
            a->fitness += gain;
            r->collected = 1;
            
            // Deposit stronger pheromone when collecting
            deposit_pheromone(pheromones, grid_size, r->x, r->y, a->arch, 2.0f);
        }
        
        // Communicate role: broadcast location
        if (min_dist < comm_strength) {
            // In stigmergy system, communication is augmented by pheromones
            deposit_pheromone(pheromones, grid_size, r->x, r->y, a->arch, 1.5f);
        }
    }
    
    // Defend role: perturbation resistance
    if (tick_num % 100 == 0 && lcgf(&a->rng) < 0.1f) {
        // Energy perturbation
        if (defend_strength < 0.3f) {
            a->energy *= 0.5f; // Halve energy if not defensive
        }
    }
    
    // Coupling with other agents
    int sample_count = min(16, num_agents/64);
    for (int s = 0; s < sample_count; s++) {
        int j = (int)(lcgf(&a->rng) * num_agents);
        if (j == idx) continue;
        
        Agent* other = &agents[j];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 0.1f) {
            float coupling = (a->arch == other->arch) ? 0.02f : 0.002f;
            
            // Role coupling
            for (int i = 0; i < 4; i++) {
                float diff = other->role[i] - a->role[i];
                a->role[i] += diff * coupling;
            }
            
            // Renormalize
            float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
            for (int i = 0; i < 4; i++) {
                a->role[i] /= sum;
            }
        }
    }
}

// Reset resources periodically
__global__ void reset_resources(Resource* resources, int num_res, unsigned int seed, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_res) return;
    
    Resource* r = &resources[idx];
    unsigned int rng = seed + idx * 7919 + tick_num;
    
    // Respawn collected resources every 50 ticks
    if (r->collected && (tick_num % 50 == 0)) {
        r->x = (lcg(&rng) / 4294967296.0f) * WORLD_SIZE;
        r->y = (lcg(&rng) / 4294967296.0f) * WORLD_SIZE;
        r->value = 0.8f + (lcg(&