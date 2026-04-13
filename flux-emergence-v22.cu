// CUDA Simulation Experiment v22: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist efficiency
// Prediction: Pheromones will help uniform agents more than specialists (reducing specialist advantage)
// because specialists already have optimized roles, while uniform agents benefit more from shared information.
// Baseline: v8 confirmed mechanisms (scarcity, territory, comms) + anti-convergence
// Novel: Agents leave pheromone markers at collected resource sites that decay over time

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
    int collected;        // Collection flag
};

// Pheromone structure for 2D grid
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

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int num_agents, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent* a = &agents[idx];
    a->rng = seed + idx * 137;
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    // Specialized group (first half) vs uniform control (second half)
    if (idx < num_agents / 2) {
        // Specialized: role[arch] = 0.7, others 0.1
        a->arch = idx % ARCH_COUNT;
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform: all roles = 0.25
        a->arch = ARCH_EXPLORER;
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources, int num_res, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_res) return;
    
    unsigned int rng = seed + idx * 7919;
    resources[idx].x = lcgf(&rng);
    resources[idx].y = lcgf(&rng);
    resources[idx].value = 0.5f + lcgf(&rng) * 0.5f; // 0.5-1.0
    resources[idx].collected = 0;
}

// Clear pheromone grid kernel
__global__ void clear_pheromones(PheromoneGrid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    grid->trail[idx] = 0.0f;
}

// Decay pheromones kernel
__global__ void decay_pheromones(PheromoneGrid* grid, float decay_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    grid->trail[idx] *= decay_rate;
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, PheromoneGrid* pheromones, 
                     int num_agents, int num_res, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = (int)(lcgf(&a->rng) * num_agents);
    if (other_idx >= num_agents) other_idx = num_agents - 1;
    Agent* other = &agents[other_idx];
    
    float similarity = 0.0f;
    for (int i = 0; i < ARCH_COUNT; i++) {
        similarity += fabsf(a->role[i] - other->role[i]);
    }
    similarity = 1.0f - similarity / ARCH_COUNT;
    
    if (similarity > 0.9f) {
        // Apply random drift to non-dominant role
        int drift_role;
        do {
            drift_role = (int)(lcgf(&a->rng) * ARCH_COUNT);
        } while (drift_role == a->arch);
        
        float drift = lcgf(&a->rng) * 0.02f - 0.01f;
        a->role[drift_role] += drift;
        a->role[drift_role] = fmaxf(0.05f, fminf(0.8f, a->role[drift_role]));
    }
    
    // Normalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < ARCH_COUNT; i++) {
        a->role[i] /= sum;
    }
    
    // Update velocity based on roles and pheromones
    float explore_strength = a->role[ARCH_EXPLORER];
    float collect_strength = a->role[ARCH_COLLECTOR];
    float comm_strength = a->role[ARCH_COMMUNICATOR];
    float defend_strength = a->role[ARCH_DEFENDER];
    
    // Random exploration component
    a->vx += (lcgf(&a->rng) - 0.5f) * 0.02f * explore_strength;
    a->vy += (lcgf(&a->rng) - 0.5f) * 0.02f * explore_strength;
    
    // Pheromone following: move toward higher pheromone concentrations
    int cell_x = (int)(a->x / CELL_SIZE);
    int cell_y = (int)(a->y / CELL_SIZE);
    cell_x = max(0, min(PHEROMONE_GRID - 1, cell_x));
    cell_y = max(0, min(PHEROMONE_GRID - 1, cell_y));
    
    // Sample pheromone gradient
    float center = pheromones->trail[cell_y * PHEROMONE_GRID + cell_x];
    float right = (cell_x < PHEROMONE_GRID - 1) ? 
                  pheromones->trail[cell_y * PHEROMONE_GRID + (cell_x + 1)] : center;
    float left = (cell_x > 0) ? 
                 pheromones->trail[cell_y * PHEROMONE_GRID + (cell_x - 1)] : center;
    float up = (cell_y < PHEROMONE_GRID - 1) ? 
               pheromones->trail[(cell_y + 1) * PHEROMONE_GRID + cell_x] : center;
    float down = (cell_y > 0) ? 
                 pheromones->trail[(cell_y - 1) * PHEROMONE_GRID + cell_x] : center;
    
    // Move toward higher pheromones (weighted by collect role)
    float dx = (right - left) * 0.5f;
    float dy = (up - down) * 0.5f;
    a->vx += dx * 0.01f * collect_strength;
    a->vy += dy * 0.01f * collect_strength;
    
    // Velocity damping and bounds
    float speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
    if (speed > 0.03f) {
        a->vx *= 0.03f / speed;
        a->vy *= 0.03f / speed;
    }
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // World wrap-around
    if (a->x < 0) a->x += WORLD_SIZE;
    if (a->x >= WORLD_SIZE) a->x -= WORLD_SIZE;
    if (a->y < 0) a->y += WORLD_SIZE;
    if (a->y >= WORLD_SIZE) a->y -= WORLD_SIZE;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    for (int i = 0; i < num_res; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        if (dx > 0.5f) dx -= WORLD_SIZE;
        if (dx < -0.5f) dx += WORLD_SIZE;
        if (dy > 0.5f) dy -= WORLD_SIZE;
        if (dy < -0.5f) dy += WORLD_SIZE;
        
        float dist = sqrtf(dx * dx + dy * dy);
        
        // Detection range based on explore role
        float detect_range = 0.03f + 0.04f * explore_strength;
        
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    if (best_res != -1) {
        Resource* r = &resources[best_res];
        
        // Collection range based on collect role
        float grab_range = 0.02f + 0.02f * a->role[ARCH_COLLECTOR];
        
        if (best_dist < grab_range) {
            // Collect resource
            float value = r->value;
            
            // Collector bonus
            value *= (1.0f + 0.5f * a->role[ARCH_COLLECTOR]);
            
            // Defender territory bonus
            int nearby_defenders = 0;
            for (int j = 0; j < 16; j++) {
                int neighbor_idx = (idx + j * 67) % num_agents;
                if (neighbor_idx == idx) continue;
                Agent* neighbor = &agents[neighbor_idx];
                
                float ndx = neighbor->x - a->x;
                float ndy = neighbor->y - a->y;
                if (ndx > 0.5f) ndx -= WORLD_SIZE;
                if (ndx < -0.5f) ndx += WORLD_SIZE;
                if (ndy > 0.5f) ndy -= WORLD_SIZE;
                if (ndy < -0.5f) ndy += WORLD_SIZE;
                
                float ndist = sqrtf(ndx * ndx + ndy * ndy);
                if (ndist < 0.06f && neighbor->arch == ARCH_DEFENDER) {
                    nearby_defenders++;
                }
            }
            value *= (1.0f + 0.2f * nearby_defenders);
            
            a->energy += value;
            a->fitness += value;
            r->collected = 1;
            
            // LEAVE PHEROMONE at collection site (NOVEL MECHANISM)
            int pcell_x = (int)(r->x / CELL_SIZE);
            int pcell_y = (int)(r->y / CELL_SIZE);
            pcell_x = max(0, min(PHEROMONE_GRID - 1, pcell_x));
            pcell_y = max(0, min(PHEROMONE_GRID - 1, pcell_y));
            
            float* pheromone = &pheromones->trail[pcell_y * PHEROMONE_GRID + pcell_x];
            atomicAdd(pheromone, 0.5f * value); // Stronger pheromone for higher value resources
        }
        
        // Communication: broadcast location to nearby agents
        if (a->role[ARCH_COMMUNICATOR] > 0.3f) {
            for (int j = 0; j < 8; j++) {
                int neighbor_idx = (idx + j * 113) % num_agents;
                if (neighbor_idx == idx) continue;
                
                Agent* neighbor = &agents[neighbor_idx];
                float ndx = neighbor->x - a->x;
                float ndy = neighbor->y - a->y;
                if (ndx > 0.5f) ndx -= WORLD_SIZE;
                if (ndx < -0.5f) ndx += WORLD_SIZE;
                if (ndy > 0.5f) ndy -= WORLD_SIZE;
                if (ndy < -0.5f) ndy += WORLD_SIZE;
                
                float ndist = sqrtf(ndx * ndx + ndy * ndy);
                if (ndist < 0.06f) {
                    // Influence neighbor's velocity toward resource
                    float influence = 0.01f * a->role[ARCH_COMMUNICATOR];
                    neighbor->vx += (r->x - neighbor->x) * influence;
                    neighbor->vy += (r->y - neighbor->y) * influence;
                }
            }
        }
    }
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick_num % 50 == 0 && idx % 17 == 0) {
        if (a->role[ARCH_DEFENDER] < 0.3f) {
            a->energy *= 0.5f;
        } else {
            a->energy *= 0.8f; // Defenders resist better
        }
    }
    
    // Coupling: adjust roles toward similar archetypes
    if (other_idx != idx) {
        float coupling = (a->arch == other->arch) ? 0.02f : 0.002f;
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] += (other->role[i] - a->role[i]) * coupling;
        }
    }
}

int main() {
    // Allocate host memory
    Agent* h_agents = new Agent[AGENTS];
    Resource* h_resources = new Resource[RESOURCES];
    
    // Allocate device memory
    Agent* d_agents;
    Resource* d_resources;
    PheromoneGrid* d_pheromones;
    
    cudaMalloc(&d_agents, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    cudaMalloc(&d_pheromones, sizeof(PheromoneGrid));
    
    // Initialize
    dim3 block(256);
    dim3 grid_agents((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_phero((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    init_agents<<<grid_agents, block>>>(d_agents, AGENTS, 12345);
    init_resources<<<grid_res, block>>>(d_resources, RESOURCES, 67890);
    clear_pheromones<<<grid_phero, block>>>(d_pheromones);
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int tick_num = 0; tick_num < TICKS; tick_num++) {
        // Decay pheromones each tick
        decay_pheromones<<<grid_phero, block>>>(d_pheromones, 0.95f);
        
        // Main simulation tick
        tick<<<grid_agents, block>>>(d_agents, d_resources, d_pheromones, 
                                     AGENTS, RESOURCES, tick_num);
        
        // Respawn resources every 50 ticks
        if (tick_num % 50 == 0 && tick_num > 0) {
            init_resources<<