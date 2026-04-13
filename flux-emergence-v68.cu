// CUDA Simulation Experiment v68: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist efficiency
// Prediction: Pheromones will help uniform agents more than specialists (falsifying hypothesis)
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Novelty: Agents leave pheromone markers that decay over time, others can follow gradients

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 256; // 256x256 grid for pheromone field
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
    int arch;             // Dominant archetype (0-3)
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
    float trail[PHEROMONE_GRID][PHEROMONE_GRID];
};

// Linear Congruential Generator (device + host)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
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
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
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
    
    unsigned int rng = idx * 19 + 67890;
    resources[idx].x = lcgf(&rng);
    resources[idx].y = lcgf(&rng);
    resources[idx].value = 0.5f + lcgf(&rng) * 0.5f; // 0.5-1.0
    resources[idx].collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(PheromoneGrid* grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < PHEROMONE_GRID && y < PHEROMONE_GRID) {
        grid->trail[x][y] = 0.0f;
    }
}

// Decay pheromones kernel
__global__ void decay_pheromones(PheromoneGrid* grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < PHEROMONE_GRID && y < PHEROMONE_GRID) {
        grid->trail[x][y] *= 0.95f; // 5% decay per tick
    }
}

// Deposit pheromone at location
__device__ void deposit_pheromone(PheromoneGrid* grid, float x, float y, float amount) {
    int gx = min(max((int)(x / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    int gy = min(max((int)(y / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    atomicAdd(&grid->trail[gx][gy], amount);
}

// Sample pheromone at location
__device__ float sample_pheromone(PheromoneGrid* grid, float x, float y) {
    int gx = min(max((int)(x / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    int gy = min(max((int)(y / CELL_SIZE), 0), PHEROMONE_GRID - 1);
    return grid->trail[gx][gy];
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, PheromoneGrid* pheromones, 
                     int* resource_counter, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect similarity > 0.9, apply drift
    float similarity = a->role[0] * a->role[0] + a->role[1] * a->role[1] +
                       a->role[2] * a->role[2] + a->role[3] * a->role[3];
    if (similarity > 0.9f) {
        int dominant = 0;
        float max_role = a->role[0];
        for (int i = 1; i < 4; i++) {
            if (a->role[i] > max_role) {
                max_role = a->role[i];
                dominant = i;
            }
        }
        // Apply small drift to non-dominant roles
        for (int i = 0; i < 4; i++) {
            if (i != dominant) {
                a->role[i] += lcgf(&a->rng) * 0.02f - 0.01f;
            }
        }
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) a->role[i] /= sum;
    }
    
    // Movement with pheromone following
    float move_x = a->vx;
    float move_y = a->vy;
    
    // Sample pheromone gradient in 4 directions
    float px = a->x;
    float py = a->y;
    float current = sample_pheromone(pheromones, px, py);
    float right = sample_pheromone(pheromones, px + 0.01f, py);
    float left = sample_pheromone(pheromones, px - 0.01f, py);
    float up = sample_pheromone(pheromones, px, py + 0.01f);
    float down = sample_pheromone(pheromones, px, py - 0.01f);
    
    // Follow positive gradient (weighted by explore role)
    float pheromone_weight = a->role[0] * 0.5f; // Explorers follow pheromones more
    if (right > current) move_x += pheromone_weight * 0.005f;
    if (left > current) move_x -= pheromone_weight * 0.005f;
    if (up > current) move_y += pheromone_weight * 0.005f;
    if (down > current) move_y -= pheromone_weight * 0.005f;
    
    // Add random exploration
    move_x += (lcgf(&a->rng) - 0.5f) * 0.01f * a->role[0];
    move_y += (lcgf(&a->rng) - 0.5f) * 0.01f * a->role[0];
    
    // Update position
    a->x += move_x;
    a->y += move_y;
    
    // Wrap-around world
    if (a->x < 0) a->x += 1.0f;
    if (a->x >= 1.0f) a->x -= 1.0f;
    if (a->y < 0) a->y += 1.0f;
    if (a->y >= 1.0f) a->y -= 1.0f;
    
    // Resource interaction
    float detect_range = 0.03f + a->role[0] * 0.04f; // Explore role increases detection
    float grab_range = 0.02f + a->role[1] * 0.02f;   // Collect role increases grab
    
    // Find nearest resource
    int nearest_idx = -1;
    float nearest_dist = 1.0f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        dx = fabs(fmod(dx + 0.5f, 1.0f) - 0.5f);
        dy = fabs(fmod(dy + 0.5f, 1.0f) - 0.5f);
        float dist = sqrt(dx*dx + dy*dy);
        
        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest_idx = i;
        }
    }
    
    // If resource found within detection range
    if (nearest_idx >= 0 && nearest_dist < detect_range) {
        Resource* r = &resources[nearest_idx];
        
        // Deposit pheromone at resource location (more if collector)
        deposit_pheromone(pheromones, r->x, r->y, 0.1f + a->role[1] * 0.2f);
        
        // Collect if within grab range
        if (nearest_dist < grab_range) {
            float value = r->value;
            // Collector bonus
            value *= (1.0f + a->role[1] * 0.5f);
            
            // Territory bonus from nearby defenders of same arch
            int defenders_nearby = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent* other = &agents[j];
                if (other->arch != a->arch) continue;
                
                float odx = other->x - a->x;
                float ody = other->y - a->y;
                odx = fabs(fmod(odx + 0.5f, 1.0f) - 0.5f);
                ody = fabs(fmod(ody + 0.5f, 1.0f) - 0.5f);
                float odist = sqrt(odx*odx + ody*ody);
                
                if (odist < 0.1f && other->role[3] > 0.3f) {
                    defenders_nearby++;
                }
            }
            value *= (1.0f + defenders_nearby * 0.2f);
            
            // Collect resource
            a->energy += value;
            a->fitness += value;
            r->collected = 1;
            atomicAdd(resource_counter, 1);
            
            // Deposit stronger pheromone on collection
            deposit_pheromone(pheromones, r->x, r->y, 0.5f);
        }
        
        // Communication: broadcast location to nearby agents
        float comm_range = 0.06f;
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent* other = &agents[j];
            
            float odx = other->x - a->x;
            float ody = other->y - a->y;
            odx = fabs(fmod(odx + 0.5f, 1.0f) - 0.5f);
            ody = fabs(fmod(ody + 0.5f, 1.0f) - 0.5f);
            float odist = sqrt(odx*odx + ody*ody);
            
            if (odist < comm_range && a->role[2] > 0.3f) {
                // Attract other agent toward resource
                float dx = r->x - other->x;
                float dy = r->y - other->y;
                dx = fmod(dx + 0.5f, 1.0f) - 0.5f;
                dy = fmod(dy + 0.5f, 1.0f) - 0.5f;
                float len = sqrt(dx*dx + dy*dy);
                if (len > 0) {
                    other->vx += dx / len * 0.01f * a->role[2];
                    other->vy += dy / len * 0.01f * a->role[2];
                }
            }
        }
    }
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick_num % 50 == 0) {
        if (a->role[3] < 0.5f) { // Not a strong defender
            a->energy *= 0.5f;
        } else {
            a->energy *= 0.8f; // Defenders resist better
        }
    }
    
    // Velocity damping
    a->vx *= 0.95f;
    a->vy *= 0.95f;
}

// Reset resources periodically
__global__ void reset_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    if (resources[idx].collected) {
        unsigned int rng = idx * 19 + 67890 + 123;
        resources[idx].x = lcgf(&rng);
        resources[idx].y = lcgf(&rng);
        resources[idx].value = 0.5f + lcgf(&rng) * 0.5f;
        resources[idx].collected = 0;
    }
}

int main() {
    printf("Experiment v68: Stigmergy with Pheromone Trails\n");
    printf("Testing: Do pheromone trails help uniform agents more than specialists?\n");
    printf("Prediction: Uniform agents benefit more (falsifying hypothesis)\n\n");
    
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    PheromoneGrid* d_pheromones_spec;
    PheromoneGrid* d_pheromones_uniform;
    int* d_resource_counter_spec;
    int* d_resource_counter_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, sizeof(PheromoneGrid));
    cudaMalloc(&d_pheromones_uniform, sizeof(PheromoneGrid));
    cudaMalloc(&d_resource_counter_spec, sizeof(int));
    cudaMalloc(&d_resource_counter_uniform, sizeof(int));
    
    // Host copies for results
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    int h_counter_spec, h_counter_uniform;
    float h_fitness_spec = 0.0f, h_fitness_uniform = 0.0f;
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID + 15) / 16, (PHEROMONE_GRID + 15) / 16);
    dim3 block_ph(16, 16);
    
    // Specialized population
    init_agents<<<