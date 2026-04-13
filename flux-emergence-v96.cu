/*
CUDA Simulation Experiment v96: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination beyond v8's 1.61x advantage,
            as specialists can follow trails to resources more efficiently.
            Expected ratio: 1.8x-2.0x for specialists vs uniform.
Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence.
Novel: Pheromone trails with spatial diffusion and decay.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 256; // Spatial grid for pheromone field
const float WORLD_SIZE = 1.0f;
const float PHEROMONE_DECAY = 0.95f;
const float DIFFUSION_RATE = 0.1f;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent struct
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Dominant archetype
    unsigned int rng;     // RNG state
};

// Resource struct
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    bool collected;       // Collection status
};

// Pheromone grid (device global memory)
__device__ float pheromone[PHEROMONE_GRID][PHEROMONE_GRID];

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
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    // Specialized vs uniform control (first half specialized, second half uniform)
    if (idx < num_agents / 2) {
        // Specialized: role[arch] = 0.7, others 0.1
        a->arch = idx % 4;
        for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
        a->role[a->arch] = 0.7f;
    } else {
        // Uniform: all roles 0.25
        a->arch = idx % 4;
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources, int num_res, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_res) return;
    
    Resource* r = &resources[idx];
    unsigned int rng = seed + idx * 7919;
    r->x = lcgf(&rng);
    r->y = lcgf(&rng);
    r->value = 0.5f + lcgf(&rng) * 0.5f; // 0.5-1.0
    r->collected = false;
}

// Initialize pheromone grid
__global__ void init_pheromone() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= PHEROMONE_GRID || y >= PHEROMONE_GRID) return;
    pheromone[x][y] = 0.0f;
}

// Diffuse and decay pheromone
__global__ void update_pheromone() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= PHEROMONE_GRID-1 || y < 1 || y >= PHEROMONE_GRID-1) return;
    
    // Simple diffusion (Laplacian)
    float diff = (pheromone[x-1][y] + pheromone[x+1][y] +
                  pheromone[x][y-1] + pheromone[x][y+1] -
                  4.0f * pheromone[x][y]) * DIFFUSION_RATE;
    
    // Apply diffusion and decay
    pheromone[x][y] = (pheromone[x][y] + diff) * PHEROMONE_DECAY;
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, int num_agents, int num_res, int tick_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect similarity > 0.9 and drift non-dominant roles
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) similarity += a->role[i] * a->role[i];
    similarity = sqrt(similarity);
    if (similarity > 0.9f) {
        int dominant = 0;
        for (int i = 1; i < 4; i++) if (a->role[i] > a->role[dominant]) dominant = i;
        int drift_target = (dominant + 1 + (a->rng % 3)) % 4;
        a->role[drift_target] += 0.01f;
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) a->role[i] /= sum;
    }
    
    // Movement based on roles
    float explore_strength = a->role[0];
    float collect_strength = a->role[1];
    float comm_strength = a->role[2];
    float defend_strength = a->role[3];
    
    // Explore: random walk
    a->vx += (lcgf(&a->rng) - 0.5f) * 0.002f * explore_strength;
    a->vy += (lcgf(&a->rng) - 0.5f) * 0.002f * explore_strength;
    
    // Velocity damping and bounds
    a->vx *= 0.95f;
    a->vy *= 0.95f;
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0) { a->x = 0; a->vx = -a->vx; }
    if (a->x >= WORLD_SIZE) { a->x = WORLD_SIZE - 0.001f; a->vx = -a->vx; }
    if (a->y < 0) { a->y = 0; a->vy = -a->vy; }
    if (a->y >= WORLD_SIZE) { a->y = WORLD_SIZE - 0.001f; a->vy = -a->vy; }
    
    // Pheromone sensing (novel mechanism)
    int px = (int)(a->x * PHEROMONE_GRID);
    int py = (int)(a->y * PHEROMONE_GRID);
    px = max(0, min(PHEROMONE_GRID-1, px));
    py = max(0, min(PHEROMONE_GRID-1, py));
    
    // Move toward higher pheromone (collectors and explorers sense more strongly)
    float sense_weight = (collect_strength + explore_strength) * 0.5f;
    if (sense_weight > 0.1f && px > 0 && px < PHEROMONE_GRID-1 && py > 0 && py < PHEROMONE_GRID-1) {
        float dx = (pheromone[px+1][py] - pheromone[px-1][py]) * 0.0005f * sense_weight;
        float dy = (pheromone[px][py+1] - pheromone[px][py-1]) * 0.0005f * sense_weight;
        a->vx += dx;
        a->vy += dy;
    }
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    for (int i = 0; i < num_res; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrt(dx*dx + dy*dy);
        
        // Detection range based on explore role
        if (dist < 0.03f + 0.04f * explore_strength && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    if (best_res >= 0) {
        Resource* r = &resources[best_res];
        
        // Collect if in range (collect role increases range and bonus)
        float grab_range = 0.02f + 0.02f * collect_strength;
        if (best_dist < grab_range) {
            float bonus = 1.0f + 0.5f * collect_strength; // Up to 50% bonus
            a->energy += r->value * bonus;
            a->fitness += r->value * bonus;
            r->collected = true;
            
            // NOVEL: Leave pheromone at collected resource location
            int pheromone_x = (int)(r->x * PHEROMONE_GRID);
            int pheromone_y = (int)(r->y * PHEROMONE_GRID);
            if (pheromone_x >= 0 && pheromone_x < PHEROMONE_GRID &&
                pheromone_y >= 0 && pheromone_y < PHEROMONE_GRID) {
                atomicAdd(&pheromone[pheromone_x][pheromone_y], 1.0f);
            }
        }
        
        // Communicate location to nearby agents (communication role)
        if (comm_strength > 0.1f) {
            float comm_range = 0.06f;
            for (int j = 0; j < num_agents; j++) {
                if (j == idx) continue;
                Agent* other = &agents[j];
                float dx = other->x - a->x;
                float dy = other->y - a->y;
                if (dx*dx + dy*dy < comm_range*comm_range) {
                    // Influence other's velocity toward resource
                    float influence = 0.001f * comm_strength;
                    other->vx += (r->x - other->x) * influence;
                    other->vy += (r->y - other->y) * influence;
                }
            }
        }
    }
    
    // Territory defense (defenders boost nearby same-arch agents)
    if (defend_strength > 0.1f) {
        float defend_range = 0.04f;
        int defender_count = 1; // self
        
        for (int j = 0; j < num_agents; j++) {
            if (j == idx) continue;
            Agent* other = &agents[j];
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            if (dx*dx + dy*dy < defend_range*defend_range && other->arch == a->arch) {
                defender_count++;
            }
        }
        
        // Defense boost: 20% per nearby defender
        float boost = 1.0f + 0.2f * (defender_count - 1);
        a->energy *= boost;
        
        // Perturbation resistance (defenders lose less energy from perturbations)
        if (tick_id % 100 == 0 && defend_strength < 0.5f) {
            a->energy *= 0.5f; // Perturbation
        }
    }
    
    // Coupling: align with same archetype, diverge from different
    float align_range = 0.05f;
    for (int j = 0; j < num_agents; j++) {
        if (j == idx) continue;
        Agent* other = &agents[j];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        float dist2 = dx*dx + dy*dy;
        if (dist2 < align_range*align_range) {
            float coupling = (a->arch == other->arch) ? 0.02f : 0.002f;
            a->vx += coupling * (other->vx - a->vx);
            a->vy += coupling * (other->vy - a->vy);
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
    cudaMalloc(&d_agents, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    
    // Initialize
    unsigned int seed = 123456789;
    dim3 block(256);
    dim3 grid_agents((AGENTS + block.x - 1) / block.x);
    dim3 grid_res((RESOURCES + block.x - 1) / block.x);
    dim3 grid_phero(PHEROMONE_GRID/16, PHEROMONE_GRID/16);
    dim3 block_phero(16, 16);
    
    init_agents<<<grid_agents, block>>>(d_agents, AGENTS, seed);
    init_resources<<<grid_res, block>>>(d_resources, RESOURCES, seed + 1);
    init_pheromone<<<grid_phero, block_phero>>>();
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        // Update pheromone field
        update_pheromone<<<grid_phero, block_phero>>>();
        
        // Run agent tick
        tick<<<grid_agents, block>>>(d_agents, d_resources, AGENTS, RESOURCES, t);
        
        // Respawn resources every 50 ticks (scarcity mechanism)
        if (t % 50 == 49) {
            init_resources<<<grid_res, block>>>(d_resources, RESOURCES, seed + t + 2);
        }
        
        cudaDeviceSynchronize();
    }
    
    // Copy results back
    cudaMemcpy(h_agents, d_agents, sizeof(Agent) * AGENTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resources, d_resources, sizeof(Resource) * RESOURCES, cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float spec_fitness = 0.0f, unif_fitness = 0.0f;
    float spec_energy = 0.0f, unif_energy = 0.0f;
    int spec_count = AGENTS / 2;
    int unif_count = AGENTS / 2;
    
    for (int i = 0; i < AGENTS; i++) {
        if (i < spec_count) {
            spec_fitness += h_agents[i].fitness;
            spec_energy += h_agents[i].energy;
        } else {
            unif_fitness += h_agents[i].fitness;
            unif_energy += h_agents[i].energy;
        }
    }
    
    spec_fitness /= spec_count;
    unif_fitness /= unif_count;
    spec_energy /= spec_count;
    unif_energy /= unif_count;
    
    // Calculate specialization metric
    float avg_specialization = 0.0f;
    for (int i = 0; i < spec_count; i++) {
        float max_role = 0.0f;
        for (int j = 0; j < 4; j++) {
            if (h_agents[i].role[j] > max_role) max_role = h_agents[i].role[j];
        }
       