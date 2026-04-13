// CUDA Simulation Experiment v71: STIGMERGY TRAILS
// Testing: Agents leave pheromone trails at resource locations that decay over time
// Prediction: Pheromones will improve specialist efficiency by 20-30% over baseline v8
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
const float WORLD_SIZE = 1.0f;
const float MIN_DIST = 0.0001f;

// Pheromone constants (NOVEL MECHANISM)
const int PHEROMONE_GRID = 64; // 64x64 grid
const float PHEROMONE_DECAY = 0.95f; // per tick
const float PHEROMONE_STRENGTH = 0.5f;
const float PHEROMONE_INFLUENCE = 0.03f; // agents sense pheromones within this radius

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent structure
struct Agent {
    float x, y;           // position
    float vx, vy;         // velocity
    float energy;         // energy level
    float role[4];        // behavioral roles
    float fitness;        // fitness score
    int arch;             // archetype
    unsigned int rng;     // random state
};

// Resource structure
struct Resource {
    float x, y;           // position
    float value;          // resource value
    int collected;        // collection flag
};

// Pheromone grid (NOVEL MECHANISM)
struct PheromoneGrid {
    float trail[PHEROMONE_GRID * PHEROMONE_GRID];
};

// Linear congruential generator
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Distance squared
__device__ float dist2(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx*dx + dy*dy;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    a->rng = idx * 123456789 + 1;
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    if (specialized) {
        // Specialized agents: dominant role based on archetype
        a->arch = idx % 4;
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.1f;
        }
        a->role[a->arch] = 0.7f;
    } else {
        // Uniform control: all roles equal
        a->arch = idx % 4;
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = idx * 987654321 + 1;
    resources[idx].x = lcgf(&rng);
    resources[idx].y = lcgf(&rng);
    resources[idx].value = 0.8f + lcgf(&rng) * 0.4f;
    resources[idx].collected = 0;
}

// Initialize pheromone grid (NOVEL MECHANISM)
__global__ void init_pheromones(PheromoneGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    grid->trail[idx] = 0.0f;
}

// Decay pheromones (NOVEL MECHANISM)
__global__ void decay_pheromones(PheromoneGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    grid->trail[idx] *= PHEROMONE_DECAY;
}

// Add pheromone at resource location (NOVEL MECHANISM)
__device__ void add_pheromone(PheromoneGrid* grid, float x, float y) {
    int gx = (int)(x * PHEROMONE_GRID);
    int gy = (int)(y * PHEROMONE_GRID);
    gx = max(0, min(PHEROMONE_GRID-1, gx));
    gy = max(0, min(PHEROMONE_GRID-1, gy));
    int idx = gy * PHEROMONE_GRID + gx;
    atomicAdd(&grid->trail[idx], PHEROMONE_STRENGTH);
}

// Sense pheromone gradient (NOVEL MECHANISM)
__device__ void sense_pheromone(PheromoneGrid* grid, float x, float y, float* dx, float* dy) {
    int gx = (int)(x * PHEROMONE_GRID);
    int gy = (int)(y * PHEROMONE_GRID);
    
    *dx = 0.0f;
    *dy = 0.0f;
    
    // Sample 3x3 neighborhood
    for (int dy_off = -1; dy_off <= 1; dy_off++) {
        for (int dx_off = -1; dx_off <= 1; dx_off++) {
            int sx = gx + dx_off;
            int sy = gy + dy_off;
            if (sx >= 0 && sx < PHEROMONE_GRID && sy >= 0 && sy < PHEROMONE_GRID) {
                int idx = sy * PHEROMONE_GRID + sx;
                float weight = grid->trail[idx];
                *dx += weight * dx_off;
                *dy += weight * dy_off;
            }
        }
    }
    
    // Normalize
    float mag = sqrtf(*dx * *dx + *dy * *dy);
    if (mag > 0.0f) {
        *dx /= mag;
        *dy /= mag;
    }
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, PheromoneGrid* grid, 
                     float* total_fitness, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = (idx + 37) % AGENTS;
    Agent* other = &agents[other_idx];
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - other->role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant role
        int drift_role = (int)(lcgf(&a->rng) * 4.0f);
        if (drift_role == a->arch) drift_role = (drift_role + 1) % 4;
        a->role[drift_role] += (lcgf(&a->rng) * 0.02f - 0.01f);
        a->role[drift_role] = max(0.0f, min(1.0f, a->role[drift_role]));
    }
    
    // Normalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) {
        a->role[i] /= sum;
    }
    
    // Pheromone sensing (NOVEL MECHANISM - affects explorers and collectors)
    float pheromone_dx = 0.0f, pheromone_dy = 0.0f;
    if (a->role[ARCH_EXPLORER] > 0.3f || a->role[ARCH_COLLECTOR] > 0.3f) {
        sense_pheromone(grid, a->x, a->y, &pheromone_dx, &pheromone_dy);
    }
    
    // Movement with pheromone influence
    float move_strength = 0.01f;
    a->vx += (lcgf(&a->rng) * 0.004f - 0.002f) + pheromone_dx * move_strength * a->role[ARCH_EXPLORER];
    a->vy += (lcgf(&a->rng) * 0.004f - 0.002f) + pheromone_dy * move_strength * a->role[ARCH_EXPLORER];
    
    // Velocity damping
    a->vx *= 0.95f;
    a->vy *= 0.95f;
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // World boundaries
    if (a->x < 0.0f) { a->x = 0.0f; a->vx = fabsf(a->vx); }
    if (a->x > WORLD_SIZE) { a->x = WORLD_SIZE; a->vx = -fabsf(a->vx); }
    if (a->y < 0.0f) { a->y = 0.0f; a->vy = fabsf(a->vy); }
    if (a->y > WORLD_SIZE) { a->y = WORLD_SIZE; a->vy = -fabsf(a->vy); }
    
    // Resource interaction
    float best_dist2 = 1e6f;
    int best_res = -1;
    
    // Explorer detection range
    float detect_range = 0.03f + 0.04f * a->role[ARCH_EXPLORER];
    
    for (int r = 0; r < RESOURCES; r++) {
        Resource* res = &resources[r];
        if (res->collected) continue;
        
        float d2 = dist2(a->x, a->y, res->x, res->y);
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_res = r;
        }
        
        // Detection
        if (d2 < detect_range * detect_range) {
            // Collector grab range with bonus
            float grab_range = 0.02f + 0.02f * a->role[ARCH_COLLECTOR];
            if (d2 < grab_range * grab_range) {
                // Collect resource
                float bonus = 1.0f + 0.5f * a->role[ARCH_COLLECTOR];
                a->energy += res->value * bonus;
                a->fitness += res->value * bonus;
                res->collected = 1;
                
                // Add pheromone at resource location (NOVEL MECHANISM)
                add_pheromone(grid, res->x, res->y);
                
                // Defender territory bonus
                int defender_count = 0;
                for (int j = 0; j < AGENTS; j++) {
                    if (j == idx) continue;
                    Agent* other = &agents[j];
                    if (other->arch == ARCH_DEFENDER && 
                        dist2(a->x, a->y, other->x, other->y) < 0.05f * 0.05f) {
                        defender_count++;
                    }
                }
                float territory_bonus = 1.0f + 0.2f * defender_count * a->role[ARCH_DEFENDER];
                a->energy *= territory_bonus;
                a->fitness *= territory_bonus;
            }
        }
    }
    
    // Communicator broadcasting
    if (a->role[ARCH_COMMUNICATOR] > 0.3f && best_res >= 0) {
        float broadcast_range = 0.06f;
        Resource* res = &resources[best_res];
        
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent* other = &agents[j];
            if (dist2(a->x, a->y, other->x, other->y) < broadcast_range * broadcast_range) {
                // Attract neighbor toward resource
                float dx = res->x - other->x;
                float dy = res->y - other->y;
                float dist = sqrtf(dx*dx + dy*dy) + MIN_DIST;
                other->vx += dx / dist * 0.005f * a->role[ARCH_COMMUNICATOR];
                other->vy += dy / dist * 0.005f * a->role[ARCH_COMMUNICATOR];
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0) {
        float resistance = 1.0f - 0.5f * a->role[ARCH_DEFENDER];
        a->energy *= 0.5f * resistance + 0.5f;
    }
    
    // Coupling with same archetype
    for (int j = 0; j < AGENTS; j++) {
        if (j == idx) continue;
        Agent* other = &agents[j];
        float d2 = dist2(a->x, a->y, other->x, other->y);
        float coupling_range = 0.02f;
        
        if (d2 < coupling_range * coupling_range) {
            float coupling_strength = (a->arch == other->arch) ? 0.02f : 0.002f;
            for (int i = 0; i < 4; i++) {
                a->role[i] += (other->role[i] - a->role[i]) * coupling_strength;
            }
        }
    }
    
    // Track fitness
    atomicAdd(total_fitness, a->fitness);
}

// Reset resources kernel
__global__ void reset_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    if (resources[idx].collected) {
        unsigned int rng = idx * 987654321 + resources[idx].collected;
        resources[idx].x = lcgf(&rng);
        resources[idx].y = lcgf(&rng);
        resources[idx].value = 0.8f + lcgf(&rng) * 0.4f;
        resources[idx].collected = 0;
    }
}

int main() {
    printf("Experiment v71: STIGMERGY TRAILS\n");
    printf("Testing: Pheromone trails at resource locations\n");
    printf("Prediction: 20-30%% efficiency improvement for specialists\n");
    printf("Mechanisms: Scarcity + Territory + Comms + STIGMERGY\n\n");
    
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    PheromoneGrid* d_grid_spec;
    PheromoneGrid* d_grid_uniform;
    float* d_fitness_spec;
    float* d_fitness_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_grid_spec, sizeof(PheromoneGrid));
    cudaMalloc(&d_grid_uniform, sizeof(PheromoneGrid));
    cudaMalloc(&d_fitness_spec, sizeof(float));
    cudaMalloc(&d_fitness_uniform, sizeof(float));
    
    // Initialize
    dim3 block(BLOCK_SIZE);
    dim3 grid_spec((AGENTS + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_res((RESOURCES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_ph((PHEROMONE_GRID * PHEROMONE_GRID + BLOCK_SIZE - 