// CUDA Simulation Experiment v60: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist advantage
// Prediction: Pheromones will help uniform agents more than specialists (reducing ratio to ~1.3x)
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Novel: Agents leave pheromone markers at collected resources that decay over time

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
    bool collected;       // Collection status
};

// Pheromone structure for stigmergy
struct PheromoneGrid {
    float trail[PHEROMONE_GRID][PHEROMONE_GRID];
};

// Linear congruential generator (device + host)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize pheromone grid
__global__ void initPheromones(PheromoneGrid* grid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PHEROMONE_GRID * PHEROMONE_GRID) {
        int i = idx % PHEROMONE_GRID;
        int j = idx / PHEROMONE_GRID;
        grid->trail[i][j] = 0.0f;
    }
}

// Initialize agents (specialized vs uniform)
__global__ void initAgents(Agent* agents, int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    unsigned int seed = idx * 137 + 1;
    Agent* a = &agents[idx];
    
    // Position
    a->x = lcgf(&seed);
    a->y = lcgf(&seed);
    a->vx = lcgf(&seed) * 0.02f - 0.01f;
    a->vy = lcgf(&seed) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    a->rng = seed;
    
    if (specialized) {
        // Specialized agents: one dominant role
        a->arch = idx % 4;  // Even distribution of archetypes
        
        for (int i = 0; i < 4; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform agents: all roles equal
        a->arch = -1;
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void initResources(Resource* resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    unsigned int seed = idx * 7919 + tick_num;
    Resource* r = &resources[idx];
    
    r->x = lcgf(&seed);
    r->y = lcgf(&seed);
    r->value = 0.5f + lcgf(&seed) * 0.5f;  // 0.5-1.0
    r->collected = false;
}

// Update pheromone decay and diffusion
__global__ void updatePheromones(PheromoneGrid* grid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    int i = idx % PHEROMONE_GRID;
    int j = idx / PHEROMONE_GRID;
    
    // Decay
    grid->trail[i][j] *= 0.95f;
    
    // Simple diffusion (average with neighbors)
    if (i > 0 && i < PHEROMONE_GRID-1 && j > 0 && j < PHEROMONE_GRID-1) {
        float sum = grid->trail[i][j] * 0.5f;
        sum += grid->trail[i-1][j] * 0.125f;
        sum += grid->trail[i+1][j] * 0.125f;
        sum += grid->trail[i][j-1] * 0.125f;
        sum += grid->trail[i][j+1] * 0.125f;
        grid->trail[i][j] = sum;
    }
}

// Add pheromone at location
__device__ void addPheromone(PheromoneGrid* grid, float x, float y, float amount) {
    int xi = (int)(x / CELL_SIZE);
    int yi = (int)(y / CELL_SIZE);
    
    if (xi >= 0 && xi < PHEROMONE_GRID && yi >= 0 && yi < PHEROMONE_GRID) {
        atomicAdd(&grid->trail[xi][yi], amount);
    }
}

// Sample pheromone at location
__device__ float samplePheromone(PheromoneGrid* grid, float x, float y) {
    int xi = (int)(x / CELL_SIZE);
    int yi = (int)(y / CELL_SIZE);
    
    if (xi >= 0 && xi < PHEROMONE_GRID && yi >= 0 && yi < PHEROMONE_GRID) {
        return grid->trail[xi][yi];
    }
    return 0.0f;
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, PheromoneGrid* pheromones, 
                     int tick_num, int* resource_counter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: prevent role homogenization
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - 0.25f);
    }
    similarity = 1.0f - similarity / 1.5f;
    
    if (similarity > 0.9f) {
        // Find non-dominant role
        int non_dom = 0;
        float max_role = a->role[0];
        for (int i = 1; i < 4; i++) {
            if (a->role[i] > max_role) {
                max_role = a->role[i];
                non_dom = i;
            }
        }
        non_dom = (non_dom + 1) % 4;
        
        // Apply random drift
        a->role[non_dom] += lcgf(&a->rng) * 0.02f - 0.01f;
        
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) {
            a->role[i] /= sum;
        }
    }
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick_num % 50 == 0) {
        float resistance = a->role[3] * 2.0f;  // Defenders resist
        if (lcgf(&a->rng) > resistance) {
            a->energy *= 0.5f;
            a->vx += lcgf(&a->rng) * 0.1f - 0.05f;
            a->vy += lcgf(&a->rng) * 0.1f - 0.05f;
        }
    }
    
    // Movement with pheromone influence
    float move_explore = a->role[0];
    float move_pheromone = samplePheromone(pheromones, a->x, a->y) * 2.0f;
    
    // Blend random exploration with pheromone following
    a->vx = a->vx * 0.8f + (lcgf(&a->rng) * 0.04f - 0.02f) * move_explore;
    a->vy = a->vy * 0.8f + (lcgf(&a->rng) * 0.04f - 0.02f) * move_explore;
    
    // Pheromone gradient following (if strong enough)
    if (move_pheromone > 0.1f) {
        float px = a->x + 0.01f;
        float py = a->y + 0.01f;
        float grad_x = samplePheromone(pheromones, px, a->y) - 
                      samplePheromone(pheromones, a->x - 0.01f, a->y);
        float grad_y = samplePheromone(pheromones, a->x, py) - 
                      samplePheromone(pheromones, a->x, a->y - 0.01f);
        
        a->vx += grad_x * 0.5f * move_pheromone;
        a->vy += grad_y * 0.5f * move_pheromone;
    }
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary wrap
    if (a->x < 0) a->x = 0;
    if (a->x > WORLD_SIZE) a->x = WORLD_SIZE;
    if (a->y < 0) a->y = 0;
    if (a->y > WORLD_SIZE) a->y = WORLD_SIZE;
    
    // Resource detection and collection
    float detect_range = 0.03f + a->role[0] * 0.04f;  // Explore role increases detection
    float grab_range = 0.02f + a->role[1] * 0.02f;    // Collect role increases grab
    
    // Find nearest resource
    int nearest_idx = -1;
    float nearest_dist = 1.0f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range && dist < nearest_dist) {
            nearest_dist = dist;
            nearest_idx = i;
        }
    }
    
    // Collect resource if in range
    if (nearest_idx != -1 && nearest_dist < grab_range) {
        Resource* r = &resources[nearest_idx];
        
        // Collection bonus from collect role
        float bonus = 1.0f + a->role[1] * 0.5f;
        
        // Territory bonus from nearby defenders (same archetype)
        float territory_bonus = 1.0f;
        int defender_count = 0;
        
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent* other = &agents[j];
            
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < 0.1f && other->arch == a->arch && other->role[3] > 0.3f) {
                defender_count++;
            }
        }
        
        territory_bonus += defender_count * 0.2f;
        
        // Energy gain
        float gain = r->value * bonus * territory_bonus;
        a->energy += gain;
        a->fitness += gain;
        
        // Mark resource as collected
        r->collected = true;
        atomicAdd(resource_counter, 1);
        
        // STIGMERRY: Leave pheromone trail at collected resource location
        addPheromone(pheromones, r->x, r->y, 1.0f + a->role[1] * 2.0f);
    }
    
    // Communication behavior
    if (a->role[2] > 0.3f && nearest_idx != -1) {
        Resource* r = &resources[nearest_idx];
        
        // Broadcast to nearby agents
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent* other = &agents[j];
            
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < 0.06f) {
                // Influence neighbor's movement toward resource
                float influence = a->role[2] * 0.5f;
                float dir_x = r->x - other->x;
                float dir_y = r->y - other->y;
                float len = sqrtf(dir_x*dir_x + dir_y*dir_y);
                
                if (len > 0.001f) {
                    other->vx += (dir_x / len) * 0.01f * influence;
                    other->vy += (dir_y / len) * 0.01f * influence;
                }
            }
        }
    }
    
    // Energy limits
    if (a->energy > 2.0f) a->energy = 2.0f;
    if (a->energy < 0) a->energy = 0;
}

// Reset resources periodically
__global__ void resetResources(Resource* resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    unsigned int seed = idx * 7919 + tick_num;
    Resource* r = &resources[idx];
    
    if (r->collected) {
        r->x = lcgf(&seed);
        r->y = lcgf(&seed);
        r->value = 0.5f + lcgf(&seed) * 0.5f;
        r->collected = false;
    }
}

int main() {
    printf("Experiment v60: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone trails left at resource locations\n");
    printf("Prediction: Helps uniform agents more, reducing specialist ratio to ~1.3x\n");
    printf("Baseline: v8 mechanisms (scarcity, territory, comms, anti-convergence)\n\n");
    
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    PheromoneGrid* d_pheromones_spec;
    PheromoneGrid* d_pheromones_uniform;
    int* d_resource_counter;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, sizeof(PheromoneGrid));
    cudaMalloc(&d_pheromones_uniform, sizeof(PheromoneGrid));
    cudaMalloc(&d_resource_counter, sizeof(int));
    
    // Host memory for results
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    int h_resource_counter[2] = {0, 0};
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    // Initialize specialized population
    initAgents<<<grid_spec, block>>>(d_agents_spec, 1);
    initPheromones<<<grid_ph, block>>>(d_pheromones_spec);
    
    // Initialize uniform population  
    initAgents<<