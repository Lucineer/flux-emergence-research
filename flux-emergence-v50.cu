
/*
CUDA Simulation Experiment v50: STIGMERRY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents by providing persistent environmental memory.
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence)
Novelty: Pheromone trails with spatial diffusion and decay
Comparison: Specialized archetypes (role[arch]=0.7) vs uniform control (all roles=0.25)
Expected: Specialists should show >1.61x advantage due to improved resource location memory.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const int PHEROMONE_GRID = 256; // Spatial grid for pheromone field
const float WORLD_SIZE = 1.0f;
const float CELL_SIZE = WORLD_SIZE / PHEROMONE_GRID;

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Archetype 0-3
    unsigned int rng;     // Random number state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone structure (stored in grid)
struct PheromoneGrid {
    float trail[PHEROMONE_GRID][PHEROMONE_GRID];
};

// Linear congruential generator
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent *agents, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    a.rng = seed + idx * 17;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = lcgf(a.rng) * 0.02f - 0.01f;
    a.vy = lcgf(a.rng) * 0.02f - 0.01f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % ARCHETYPES;
    
    // Specialized archetypes (70% dominant role)
    for (int i = 0; i < 4; i++) {
        a.role[i] = 0.1f;
    }
    a.role[a.arch] = 0.7f;
    
    // Uniform control group (last 512 agents)
    if (idx >= AGENTS/2) {
        for (int i = 0; i < 4; i++) {
            a.role[i] = 0.25f;
        }
        a.arch = 4; // Mark as uniform
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource *resources, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = resources[idx];
    unsigned int rng = seed + idx * 29;
    r.x = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
    r.y = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
    r.value = 0.5f + lcgf(rng) * 0.5f;
    r.collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(PheromoneGrid *grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= PHEROMONE_GRID || y >= PHEROMONE_GRID) return;
    
    grid->trail[x][y] = 0.0f;
}

// Diffuse and decay pheromones
__global__ void update_pheromones(PheromoneGrid *grid, PheromoneGrid *new_grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= PHEROMONE_GRID || y >= PHEROMONE_GRID) return;
    
    float current = grid->trail[x][y];
    float sum = current * 0.6f; // Center weight
    
    // 4-neighbor diffusion
    if (x > 0) sum += grid->trail[x-1][y] * 0.1f;
    if (x < PHEROMONE_GRID-1) sum += grid->trail[x+1][y] * 0.1f;
    if (y > 0) sum += grid->trail[x][y-1] * 0.1f;
    if (y < PHEROMONE_GRID-1) sum += grid->trail[x][y+1] * 0.1f;
    
    // Decay over time
    new_grid->trail[x][y] = sum * 0.995f;
}

// Deposit pheromone at resource location
__device__ void deposit_pheromone(PheromoneGrid *grid, float x, float y, float amount) {
    int gx = min(PHEROMONE_GRID-1, max(0, (int)(x / CELL_SIZE)));
    int gy = min(PHEROMONE_GRID-1, max(0, (int)(y / CELL_SIZE)));
    atomicAdd(&grid->trail[gx][gy], amount);
}

// Sample pheromone at position
__device__ float sample_pheromone(PheromoneGrid *grid, float x, float y) {
    int gx = min(PHEROMONE_GRID-1, max(0, (int)(x / CELL_SIZE)));
    int gy = min(PHEROMONE_GRID-1, max(0, (int)(y / CELL_SIZE)));
    return grid->trail[gx][gy];
}

// Main simulation tick kernel
__global__ void tick(Agent *agents, Resource *resources, PheromoneGrid *pheromones, 
                     int tick_num, int *resource_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: detect similarity > 0.9 and apply drift
    float role_sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    if (role_sum > 0.0f) {
        float max_role = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < 4; i++) {
            if (a.role[i] > max_role) {
                max_role = a.role[i];
                max_idx = i;
            }
        }
        if (max_role / role_sum > 0.9f) {
            // Apply random drift to non-dominant role
            int drift_idx;
            do {
                drift_idx = (int)(lcgf(a.rng) * 4);
            } while (drift_idx == max_idx);
            a.role[drift_idx] += 0.01f;
        }
    }
    
    // Normalize roles
    float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    if (sum > 0.0f) {
        for (int i = 0; i < 4; i++) {
            a.role[i] /= sum;
        }
    }
    
    // Movement with pheromone guidance
    float explore_strength = a.role[0];
    float pheromone_strength = sample_pheromone(pheromones, a.x, a.y);
    
    // Blend random exploration with pheromone-following
    if (pheromone_strength > 0.1f && explore_strength > 0.3f) {
        // Sample pheromone gradient
        float px = sample_pheromone(pheromones, min(WORLD_SIZE, a.x + 0.01f), a.y);
        float py = sample_pheromone(pheromones, a.x, min(WORLD_SIZE, a.y + 0.01f));
        float mx = sample_pheromone(pheromones, max(0.0f, a.x - 0.01f), a.y);
        float my = sample_pheromone(pheromones, a.x, max(0.0f, a.y - 0.01f));
        
        a.vx += (px - mx) * 0.5f * explore_strength;
        a.vy += (py - my) * 0.5f * explore_strength;
    } else {
        // Random walk
        a.vx += lcgf(a.rng) * 0.004f - 0.002f;
        a.vy += lcgf(a.rng) * 0.004f - 0.002f;
    }
    
    // Velocity damping
    a.vx *= 0.95f;
    a.vy *= 0.95f;
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World boundaries
    if (a.x < 0.0f) { a.x = 0.0f; a.vx = -a.vx; }
    if (a.x > WORLD_SIZE) { a.x = WORLD_SIZE; a.vx = -a.vx; }
    if (a.y < 0.0f) { a.y = 0.0f; a.vy = -a.vy; }
    if (a.y > WORLD_SIZE) { a.y = WORLD_SIZE; a.vy = -a.vy; }
    
    // Resource interaction
    float collect_range = 0.02f + a.role[1] * 0.02f; // Collect role increases range
    float detect_range = 0.03f + a.role[0] * 0.04f;  // Explore role increases detection
    
    // Find nearest resource
    int nearest_idx = -1;
    float nearest_dist = 1.0f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range && dist < nearest_dist) {
            nearest_dist = dist;
            nearest_idx = i;
        }
    }
    
    // Collect resource if in range
    if (nearest_idx != -1 && nearest_dist < collect_range) {
        Resource &r = resources[nearest_idx];
        
        // Collectors get 50% bonus
        float gain = r.value * (1.0f + 0.5f * a.role[1]);
        
        // Defenders get territory boost (20% per nearby defender)
        int defenders_nearby = 0;
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent &other = agents[j];
            if (other.arch == a.arch) {
                float odx = other.x - a.x;
                float ody = other.y - a.y;
                if (sqrtf(odx*odx + ody*ody) < 0.1f && other.role[3] > 0.5f) {
                    defenders_nearby++;
                }
            }
        }
        gain *= (1.0f + 0.2f * defenders_nearby);
        
        a.energy += gain;
        a.fitness += gain;
        r.collected = 1;
        
        // Deposit pheromone at collected resource location
        deposit_pheromone(pheromones, r.x, r.y, 1.0f + a.role[1] * 2.0f);
        
        atomicAdd(resource_counter, 1);
    }
    
    // Communication role: broadcast nearest resource location
    if (a.role[2] > 0.3f && nearest_idx != -1) {
        Resource &r = resources[nearest_idx];
        // In real implementation would broadcast to nearby agents
        // Simplified: just deposit pheromone at resource location
        deposit_pheromone(pheromones, r.x, r.y, 0.5f * a.role[2]);
    }
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick_num % 50 == 0 && tick_num > 0) {
        if (a.role[3] < 0.5f) { // Not a strong defender
            a.energy *= 0.5f;
        } else {
            a.energy *= 0.8f; // Defenders resist better
        }
    }
    
    // Death and rebirth
    if (a.energy < 0.1f) {
        a.x = lcgf(a.rng);
        a.y = lcgf(a.rng);
        a.energy = 1.0f;
        a.vx = lcgf(a.rng) * 0.02f - 0.01f;
        a.vy = lcgf(a.rng) * 0.02f - 0.01f;
    }
}

// Main function
int main() {
    // Allocate host memory
    Agent *h_agents = new Agent[AGENTS];
    Resource *h_resources = new Resource[RESOURCES];
    int h_resource_counter = 0;
    
    // Allocate device memory
    Agent *d_agents;
    Resource *d_resources;
    PheromoneGrid *d_pheromones1, *d_pheromones2;
    int *d_resource_counter;
    
    cudaMalloc(&d_agents, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    cudaMalloc(&d_pheromones1, sizeof(PheromoneGrid));
    cudaMalloc(&d_pheromones2, sizeof(PheromoneGrid));
    cudaMalloc(&d_resource_counter, sizeof(int));
    
    // Initialize
    dim3 block(256);
    dim3 grid_agents((AGENTS + 255) / 256);
    dim3 grid_resources((RESOURCES + 255) / 256);
    dim3 grid_phero(PHEROMONE_GRID/16, PHEROMONE_GRID/16);
    dim3 block_phero(16, 16);
    
    init_agents<<<grid_agents, block>>>(d_agents, 12345);
    init_resources<<<grid_resources, block>>>(d_resources, 67890);
    init_pheromones<<<grid_phero, block_phero>>>(d_pheromones1);
    init_pheromones<<<grid_phero, block_phero>>>(d_pheromones2);
    
    cudaDeviceSynchronize();
    
    // Track fitness for specialists vs uniform
    float specialist_fitness = 0.0f;
    float uniform_fitness = 0.0f;
    
    printf("Starting v50: Stigmergy Trails Experiment\n");
    printf("Agents: %d (512 specialists, 512 uniform)\n", AGENTS);
    printf("Resources: %d (scarce)\n", RESOURCES);
    printf("Ticks: %d\n", TICKS);
    printf("Pheromone grid: %dx%d\n\n", PHEROMONE_GRID, PHEROMONE_GRID);
    
    // Main simulation loop
    for (int t = 0; t < TICKS; t++) {
        // Reset resource counter
        cudaMemset(d_resource_counter, 0, sizeof(int));
        
        // Reset resources every 50 ticks (scarcity cycle)
        if (t % 50 == 0) {
            init_resources<<<grid_resources, block>>>(d_resources, 67890 + t);
        }
        
        // Run tick kernel
        tick<<<grid_agents, block>>>(d_agents, d_resources, d_pheromones1, t, d_resource_counter);
        
        // Update pheromones (ping-pong buffers)
        update_pheromones<<<grid_phero, block_phero>>>(d_pheromones1, d_pheromones2);
        
        // Swap buffers
        PheromoneGrid *temp = d_pheromones1;
        d_pheromones1 = d_pheromones2;
        d_pheromones2 = temp;
        
        cudaDeviceSynchronize();
        
        // Progress indicator
        if (t % 100 == 0) {
