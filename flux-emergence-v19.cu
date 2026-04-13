
/*
CUDA Simulation Experiment v19: STIGMERGY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist efficiency by 20-30% over v8 baseline
  because specialists can follow trails to resources faster.
Baseline: v8 mechanisms (scarcity, territory, comms, anti-convergence)
Novelty: Stigmergy - agents deposit pheromone when collecting, sense gradient
Control: Uniform agents (all roles=0.25) vs Specialized (role[arch]=0.7)
Expected: Specialists should leverage trails better due to role coordination.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK = 256;
const float WORLD_SIZE = 1.0f;
const float PHEROMONE_DECAY = 0.95f;
const float DEPOSIT_STRENGTH = 0.5f;
const float SENSE_RANGE = 0.08f;

// Agent archetype roles
enum { EXPLORER, COLLECTOR, COMMUNICATOR, DEFENDER };

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];
    float fitness;
    int arch;
    unsigned int rng;
};

// Resource struct
struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone grid cell
struct Pheromone {
    float strength[4]; // One per archetype
};

// LCG RNG
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents
__global__ void init_agents(Agent* agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    unsigned int seed = idx * 17 + 12345;
    a.x = lcgf(&seed) * WORLD_SIZE;
    a.y = lcgf(&seed) * WORLD_SIZE;
    a.vx = lcgf(&seed) * 0.02f - 0.01f;
    a.vy = lcgf(&seed) * 0.02f - 0.01f;
    a.energy = 1.0f;
    a.arch = idx % 4;
    a.rng = seed;
    a.fitness = 0.0f;
    
    if (specialized) {
        // Specialized: strong in own role, weak in others
        for (int i = 0; i < 4; i++) {
            a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control
        for (int i = 0; i < 4; i++) {
            a.role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource& r = resources[idx];
    unsigned int seed = idx * 19 + 54321;
    r.x = lcgf(&seed) * WORLD_SIZE;
    r.y = lcgf(&seed) * WORLD_SIZE;
    r.value = 0.8f + lcgf(&seed) * 0.4f;
    r.collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    for (int i = 0; i < 4; i++) {
        grid[idx].strength[i] = 0.0f;
    }
}

// Decay pheromones
__global__ void decay_pheromones(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    for (int i = 0; i < 4; i++) {
        grid[idx].strength[i] *= PHEROMONE_DECAY;
    }
}

// Get grid cell from position
__device__ int get_grid_cell(float x, float y, int grid_size) {
    int gx = min(max((int)(x * grid_size), 0), grid_size - 1);
    int gy = min(max((int)(y * grid_size), 0), grid_size - 1);
    return gy * grid_size + gx;
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromones, 
                     int grid_size, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other = (idx + 37) % AGENTS;
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a.role[i] - agents[other].role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant role
        int drift_role = (int)(lcgf(&a.rng) * 4);
        while (drift_role == a.arch) {
            drift_role = (int)(lcgf(&a.rng) * 4);
        }
        a.role[drift_role] += lcgf(&a.rng) * 0.02f - 0.01f;
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int i = 0; i < 4; i++) a.role[i] /= sum;
    }
    
    // Sense pheromone gradient
    float px = a.x;
    float py = a.y;
    int cell = get_grid_cell(px, py, grid_size);
    
    float best_dir_x = 0.0f;
    float best_dir_y = 0.0f;
    float best_strength = 0.0f;
    
    // Check neighboring cells for pheromone of own archetype
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (int)(px * grid_size) + dx;
            int ny = (int)(py * grid_size) + dy;
            if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                int ncell = ny * grid_size + nx;
                float strength = pheromones[ncell].strength[a.arch];
                if (strength > best_strength) {
                    best_strength = strength;
                    best_dir_x = dx / sqrtf(dx*dx + dy*dy);
                    best_dir_y = dy / sqrtf(dx*dx + dy*dy);
                }
            }
        }
    }
    
    // Movement: combine pheromone following with random exploration
    float explore_weight = a.role[EXPLORER];
    float pheromone_weight = a.role[COLLECTOR]; // Collectors follow trails more
    
    a.vx = a.vx * 0.8f + 
           (lcgf(&a.rng) * 0.02f - 0.01f) * explore_weight +
           best_dir_x * 0.03f * pheromone_weight;
    
    a.vy = a.vy * 0.8f + 
           (lcgf(&a.rng) * 0.02f - 0.01f) * explore_weight +
           best_dir_y * 0.03f * pheromone_weight;
    
    // Limit velocity
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.03f) {
        a.vx *= 0.03f / speed;
        a.vy *= 0.03f / speed;
    }
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World wrap
    if (a.x < 0) a.x = WORLD_SIZE + a.x;
    if (a.x >= WORLD_SIZE) a.x = a.x - WORLD_SIZE;
    if (a.y < 0) a.y = WORLD_SIZE + a.y;
    if (a.y >= WORLD_SIZE) a.y = a.y - WORLD_SIZE;
    
    // Resource interaction
    float detect_range = 0.03f + a.role[EXPLORER] * 0.04f;
    float grab_range = 0.02f + a.role[COLLECTOR] * 0.02f;
    
    // Find nearest resource
    int nearest = -1;
    float nearest_dist = 999.0f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource& r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        // Wrap distance
        if (dx > WORLD_SIZE/2) dx -= WORLD_SIZE;
        if (dx < -WORLD_SIZE/2) dx += WORLD_SIZE;
        if (dy > WORLD_SIZE/2) dy -= WORLD_SIZE;
        if (dy < -WORLD_SIZE/2) dy += WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest = i;
        }
    }
    
    if (nearest != -1) {
        Resource& r = resources[nearest];
        
        // Detection
        if (nearest_dist < detect_range) {
            // Collectors get bonus
            if (nearest_dist < grab_range) {
                float bonus = 1.0f + a.role[COLLECTOR] * 0.5f;
                a.energy += r.value * bonus;
                a.fitness += r.value * bonus;
                r.collected = 1;
                
                // DEPOSIT PHEROMONE at resource location
                int pcell = get_grid_cell(r.x, r.y, grid_size);
                atomicAdd(&pheromones[pcell].strength[a.arch], DEPOSIT_STRENGTH);
            }
            
            // Communication
            if (a.role[COMMUNICATOR] > 0.3f) {
                // Broadcast to nearby agents of same archetype
                for (int i = 0; i < AGENTS; i++) {
                    if (i == idx) continue;
                    Agent& other = agents[i];
                    if (other.arch != a.arch) continue;
                    
                    float dx = other.x - a.x;
                    float dy = other.y - a.y;
                    if (dx > WORLD_SIZE/2) dx -= WORLD_SIZE;
                    if (dx < -WORLD_SIZE/2) dx += WORLD_SIZE;
                    if (dy > WORLD_SIZE/2) dy -= WORLD_SIZE;
                    if (dy < -WORLD_SIZE/2) dy += WORLD_SIZE;
                    
                    float dist = sqrtf(dx*dx + dy*dy);
                    if (dist < 0.06f) {
                        // Influence neighbor's velocity toward resource
                        float influence = a.role[COMMUNICATOR] * 0.01f;
                        other.vx += (r.x - other.x) * influence;
                        other.vy += (r.y - other.y) * influence;
                    }
                }
            }
        }
    }
    
    // Territory defense
    if (a.role[DEFENDER] > 0.3f) {
        int defenders_nearby = 0;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent& other = agents[i];
            if (other.arch != a.arch) continue;
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx > WORLD_SIZE/2) dx -= WORLD_SIZE;
            if (dx < -WORLD_SIZE/2) dx += WORLD_SIZE;
            if (dy > WORLD_SIZE/2) dy -= WORLD_SIZE;
            if (dy < -WORLD_SIZE/2) dy += WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < 0.05f && other.role[DEFENDER] > 0.3f) {
                defenders_nearby++;
            }
        }
        
        // Defense bonus: 20% per nearby defender
        float defense_bonus = 1.0f + defenders_nearby * 0.2f;
        a.energy *= defense_bonus;
        a.fitness *= defense_bonus;
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0 && tick_num > 0) {
        float resistance = a.role[DEFENDER] * 0.5f;
        if (lcgf(&a.rng) > resistance) {
            a.energy *= 0.5f;
            // Push away from current position
            a.vx += lcgf(&a.rng) * 0.1f - 0.05f;
            a.vy += lcgf(&a.rng) * 0.1f - 0.05f;
        }
    }
    
    // Respawn resources (every 50 ticks)
    if (tick_num % 50 == 0 && tick_num > 0) {
        for (int i = 0; i < RESOURCES; i++) {
            resources[i].collected = 0;
        }
    }
}

int main() {
    // Allocate host memory
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    Resource* h_resources = new Resource[RESOURCES];
    
    // Allocate device memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    Pheromone* d_pheromones_spec;
    Pheromone* d_pheromones_uniform;
    
    cudaMalloc(&d_agents_spec, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_agents_uniform, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    
    int grid_size = 64; // 64x64 pheromone grid
    int grid_cells = grid_size * grid_size;
    cudaMalloc(&d_pheromones_spec, sizeof(Pheromone) * grid_cells);
    cudaMalloc(&d_pheromones_uniform, sizeof(Pheromone) * grid_cells);
    
    // Initialize
    dim3 block(BLOCK);
    dim3 grid_spec((AGENTS + BLOCK - 1) / BLOCK);
    dim3 grid_res((RESOURCES + BLOCK - 1) / BLOCK);
    dim3 grid_pheromone((grid_cells + BLOCK - 1) / BLOCK);
    
    // Run specialized population
    init_agents<<<grid_spec, block>>>(d_agents_spec, 1);
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromones<<<grid_pheromone, block>>>(d_pheromones_spec, grid_size);
    
    cudaDeviceSynchronize();
    
    for (int t = 0; t < TICKS; t++) {
        decay_pheromones<<<grid_pheromone, block>>>(d_pheromones_spec, grid_size);
        tick<<<grid_spec, block>>>(d_agents_spec, d_resources, d_pheromones_spec, grid_size, t);
        if (t % 50 == 0) {
            cudaDeviceSynchronize();
        }
    }
    
    cudaMemcpy(h_agents_spec, d_agents_spec, sizeof(Agent) * AGENTS, cudaMemcpyDeviceToHost);
    
    // Run uniform population
    init_agents<<<grid_spec, block>>>(d_agents_uniform, 0);
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromones<<<grid_pheromone, block>>>(d_pheromones_uniform, grid_size);
    
    cudaDeviceSynchronize();
    
    for (int t = 0; t < TICKS; t++) {
        decay_pheromones<<<grid_pheromone, block>>>(d_pheromones_uniform, grid_size);
        tick<<<grid_spec, block>>>(d_agents_uniform, d_resources, d_pheromones_uniform, grid_size, t);
        if (t % 50 == 0) {
            cudaDeviceSynchronize();
        }
   