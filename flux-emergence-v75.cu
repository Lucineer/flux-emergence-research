/*
CUDA Simulation Experiment v75: Stigmergy with Pheromone Trails
Testing: Whether pheromone trails left at resource locations improve specialist efficiency
Prediction: Pheromones will help uniform agents more (they explore randomly), 
            reducing specialist advantage from 1.61x to ~1.3x
Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
Novel: Agents leave pheromone markers at collected resources that decay over time
       Agents can detect pheromone intensity and follow gradients
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
    float decay_rate;
};

// Linear Congruential Generator (device/host)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    a->rng = idx * 17 + 12345;
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->vy = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    if (specialized) {
        // Specialized agents: one dominant role (0.7), others 0.1 each
        a->arch = idx % 4;
        for (int i = 0; i < 4; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles equal
        a->arch = -1;
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = idx * 19 + 54321;
    resources[idx].x = lcgf(&rng);
    resources[idx].y = lcgf(&rng);
    resources[idx].value = 0.8f + lcgf(&rng) * 0.4f; // 0.8-1.2
    resources[idx].collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(PheromoneGrid* grid, float decay_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    grid->trail[idx] = 0.0f;
    grid->decay_rate = decay_rate;
}

// Decay pheromones kernel
__global__ void decay_pheromones(PheromoneGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    grid->trail[idx] *= grid->decay_rate;
    if (grid->trail[idx] < 0.001f) grid->trail[idx] = 0.0f;
}

// Add pheromone at position
__device__ void add_pheromone(PheromoneGrid* grid, float x, float y, float amount) {
    int gx = (int)(x * PHEROMONE_GRID);
    int gy = (int)(y * PHEROMONE_GRID);
    gx = max(0, min(PHEROMONE_GRID - 1, gx));
    gy = max(0, min(PHEROMONE_GRID - 1, gy));
    
    int idx = gy * PHEROMONE_GRID + gx;
    atomicAdd(&grid->trail[idx], amount);
}

// Sample pheromone at position
__device__ float sample_pheromone(PheromoneGrid* grid, float x, float y) {
    int gx = (int)(x * PHEROMONE_GRID);
    int gy = (int)(y * PHEROMONE_GRID);
    gx = max(0, min(PHEROMONE_GRID - 1, gx));
    gy = max(0, min(PHEROMONE_GRID - 1, gy));
    
    return grid->trail[gy * PHEROMONE_GRID + gx];
}

// Get pheromone gradient
__device__ void get_pheromone_gradient(PheromoneGrid* grid, float x, float y, float* gx, float* gy) {
    float center = sample_pheromone(grid, x, y);
    float right = sample_pheromone(grid, x + CELL_SIZE, y);
    float left = sample_pheromone(grid, x - CELL_SIZE, y);
    float up = sample_pheromone(grid, x, y + CELL_SIZE);
    float down = sample_pheromone(grid, x, y - CELL_SIZE);
    
    *gx = (right - left) / (2.0f * CELL_SIZE);
    *gy = (up - down) / (2.0f * CELL_SIZE);
    
    // Normalize
    float len = sqrtf(*gx * *gx + *gy * *gy);
    if (len > 1e-6f) {
        *gx /= len;
        *gy /= len;
    }
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, PheromoneGrid* pheromones, 
                     int tick_num, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with neighbors
    if (tick_num % 10 == 0) {
        int neighbor_idx = (idx + 1) % AGENTS;
        Agent* neighbor = &agents[neighbor_idx];
        
        float similarity = 0.0f;
        for (int i = 0; i < 4; i++) {
            similarity += fabsf(a->role[i] - neighbor->role[i]);
        }
        similarity = 1.0f - similarity / 4.0f;
        
        if (similarity > 0.9f) {
            // Find non-dominant role and apply drift
            int drift_role = 0;
            for (int i = 1; i < 4; i++) {
                if (a->role[i] < a->role[drift_role]) {
                    drift_role = i;
                }
            }
            a->role[drift_role] += (lcgf(&a->rng) - 0.5f) * 0.01f;
            
            // Renormalize
            float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
            for (int i = 0; i < 4; i++) {
                a->role[i] /= sum;
            }
        }
    }
    
    // Perturbation (10% chance every 50 ticks)
    if (tick_num % 50 == 0 && lcgf(&a->rng) < 0.1f) {
        if (a->role[3] > 0.3f) { // Defender resistance
            a->energy *= 0.8f;
        } else {
            a->energy *= 0.5f;
        }
    }
    
    // Pheromone following behavior
    float pheromone_strength = sample_pheromone(pheromones, a->x, a->y);
    if (pheromone_strength > 0.1f) {
        float pgx, pgy;
        get_pheromone_gradient(pheromones, a->x, a->y, &pgx, &pgy);
        
        // Follow pheromone gradient based on explore role
        a->vx += pgx * 0.01f * a->role[0];
        a->vy += pgy * 0.01f * a->role[0];
    }
    
    // Role-based behaviors
    float max_behavior = max(max(a->role[0], a->role[1]), max(a->role[2], a->role[3]));
    
    if (a->role[0] == max_behavior) { // Explore
        // Random walk with some persistence
        a->vx += (lcgf(&a->rng) - 0.5f) * 0.01f;
        a->vy += (lcgf(&a->rng) - 0.5f) * 0.01f;
    }
    
    // Velocity limits and position update
    float speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
    if (speed > 0.03f) {
        a->vx *= 0.03f / speed;
        a->vy *= 0.03f / speed;
    }
    
    a->x += a->vx;
    a->y += a->vy;
    
    // World boundaries (bouncing)
    if (a->x < 0.0f) { a->x = 0.0f; a->vx = fabsf(a->vx); }
    if (a->x > WORLD_SIZE) { a->x = WORLD_SIZE; a->vx = -fabsf(a->vx); }
    if (a->y < 0.0f) { a->y = 0.0f; a->vy = fabsf(a->vy); }
    if (a->y > WORLD_SIZE) { a->y = WORLD_SIZE; a->vy = -fabsf(a->vy); }
    
    // Resource collection
    if (a->role[1] > 0.3f) { // Collector role active
        for (int r = 0; r < RESOURCES; r++) {
            Resource* res = &resources[r];
            if (res->collected) continue;
            
            float dx = res->x - a->x;
            float dy = res->y - a->y;
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist < 0.04f) { // Grab range
                // Collection bonus for collectors
                float bonus = (a->role[1] > 0.5f) ? 1.5f : 1.0f;
                a->energy += res->value * bonus;
                a->fitness += res->value * bonus;
                res->collected = 1;
                
                // Leave pheromone at collected resource location
                add_pheromone(pheromones, res->x, res->y, 1.0f);
                
                // Territory boost for defenders
                if (a->role[3] > 0.3f) {
                    int nearby_defenders = 0;
                    for (int j = 0; j < AGENTS; j++) {
                        if (j == idx) continue;
                        Agent* other = &agents[j];
                        float odx = other->x - a->x;
                        float ody = other->y - a->y;
                        if (sqrtf(odx * odx + ody * ody) < 0.1f && other->role[3] > 0.3f) {
                            nearby_defenders++;
                        }
                    }
                    a->energy += res->value * 0.2f * nearby_defenders;
                    a->fitness += res->value * 0.2f * nearby_defenders;
                }
                break;
            }
        }
    }
    
    // Communication behavior
    if (a->role[2] > 0.3f && tick_num % 5 == 0) {
        // Find nearest resource
        float best_dist = 1e6f;
        float best_x = 0.0f, best_y = 0.0f;
        
        for (int r = 0; r < RESOURCES; r++) {
            Resource* res = &resources[r];
            if (res->collected) continue;
            
            float dx = res->x - a->x;
            float dy = res->y - a->y;
            float dist = dx * dx + dy * dy;
            
            if (dist < best_dist) {
                best_dist = dist;
                best_x = res->x;
                best_y = res->y;
            }
        }
        
        if (best_dist < 1e5f) {
            // Leave pheromone trail toward resource (weaker than direct collection)
            add_pheromone(pheromones, best_x, best_y, 0.3f);
        }
    }
    
    // Energy-based velocity adjustment
    if (a->energy < 0.5f) {
        a->vx *= 0.9f;
        a->vy *= 0.9f;
    }
}

// Reset resources kernel
__global__ void reset_resources(Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    // 10% chance to respawn each tick
    unsigned int rng_state = idx * 23 + tick_num;
    if (resources[idx].collected && lcgf(&rng_state) < 0.1f) {
        unsigned int rng = idx * 23 + tick_num;
        resources[idx].x = lcgf(&rng);
        resources[idx].y = lcgf(&rng);
        resources[idx].value = 0.8f + lcgf(&rng) * 0.4f;
        resources[idx].collected = 0;
    }
}

int main() {
    printf("Experiment v75: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone trails at resource locations\n");
    printf("Prediction: Reduces specialist advantage from 1.61x to ~1.3x\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RESOURCES, TICKS);
    
    // Allocate device memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    PheromoneGrid* d_pheromones_spec;
    PheromoneGrid* d_pheromones_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, sizeof(PheromoneGrid));
    cudaMalloc(&d_pheromones_uniform, sizeof(PheromoneGrid));
    
    // Host memory for results
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    init_agents<<<grid_spec, block>>>(d_agents_spec, 1); // Specialized
    init_agents<<<grid_spec, block>>>(d_agents_uniform, 0); // Uniform
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromones<<<grid