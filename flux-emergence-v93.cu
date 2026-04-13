/*
CUDA Simulation Experiment v93: STIGMERY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will improve collective efficiency for specialized agents 
            more than for uniform agents, increasing the specialist advantage ratio.
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence)
Novel: Stigmergy - agents deposit pheromone when collecting resources, 
       other agents can sense pheromone gradient to find resources.
Control: Uniform agents (all roles=0.25) vs Specialized (role[arch]=0.7)
Expected: Specialists should benefit more from stigmergy due to role coordination.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const int PHEROMONE_GRID = 256; // 256x256 grid for pheromone map
const float WORLD_SIZE = 1.0f;

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Archetype 0-3
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone structure for stigmergy
struct Pheromone {
    float value;          // Pheromone concentration
    int timestamp;        // Last update tick
};

// Linear Congruential Generator
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
    a->arch = idx % ARCHETYPES;
    
    if (specialized) {
        // Specialized agents: strong in one role based on archetype
        for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
        a->role[a->arch] = 0.7f;
    } else {
        // Uniform control: all roles equal
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = idx * 19 + 54321;
    resources[idx].x = lcgf(&rng);
    resources[idx].y = lcgf(&rng);
    resources[idx].value = 0.5f + lcgf(&rng) * 0.5f;
    resources[idx].collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    pheromones[idx].value = 0.0f;
    pheromones[idx].timestamp = 0;
}

// Update pheromone decay
__global__ void decay_pheromones(Pheromone* pheromones, int current_tick) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    Pheromone* p = &pheromones[idx];
    if (current_tick - p->timestamp > 10) { // Decay after 10 ticks
        p->value *= 0.9f;
        if (p->value < 0.001f) p->value = 0.0f;
    }
}

// Deposit pheromone at location
__device__ void deposit_pheromone(Pheromone* pheromones, float x, float y, float amount, int tick) {
    int grid_x = (int)(x * PHEROMONE_GRID);
    int grid_y = (int)(y * PHEROMONE_GRID);
    if (grid_x < 0 || grid_x >= PHEROMONE_GRID || grid_y < 0 || grid_y >= PHEROMONE_GRID) return;
    
    int idx = grid_y * PHEROMONE_GRID + grid_x;
    atomicAdd(&pheromones[idx].value, amount);
    pheromones[idx].timestamp = tick;
}

// Sample pheromone at location
__device__ float sample_pheromone(Pheromone* pheromones, float x, float y) {
    int grid_x = (int)(x * PHEROMONE_GRID);
    int grid_y = (int)(y * PHEROMONE_GRID);
    if (grid_x < 0 || grid_x >= PHEROMONE_GRID || grid_y < 0 || grid_y >= PHEROMONE_GRID) return 0.0f;
    
    int idx = grid_y * PHEROMONE_GRID + grid_x;
    return pheromones[idx].value;
}

// Main simulation kernel
__global__ void tick_kernel(Agent* agents, Resource* resources, Pheromone* pheromones, 
                           int current_tick, int* resource_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence mechanism
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - 0.25f);
    }
    similarity /= 4.0f;
    
    if (similarity > 0.9f) {
        // Find non-dominant role
        int dominant = a->arch;
        int target = (dominant + 1) % 4;
        a->role[target] += 0.01f;
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) a->role[i] /= sum;
    }
    
    // Movement with pheromone gradient sensing
    float dx = 0.0f, dy = 0.0f;
    
    // Sample pheromone in surrounding 4 directions
    float px = a->x;
    float py = a->y;
    float sense_range = 0.01f;
    
    float pheromone_center = sample_pheromone(pheromones, px, py);
    float pheromone_right = sample_pheromone(pheromones, px + sense_range, py);
    float pheromone_left = sample_pheromone(pheromones, px - sense_range, py);
    float pheromone_up = sample_pheromone(pheromones, px, py + sense_range);
    float pheromone_down = sample_pheromone(pheromones, px, py - sense_range);
    
    // Move toward higher pheromone concentration (weighted by explore role)
    dx += (pheromone_right - pheromone_left) * a->role[0] * 0.5f;
    dy += (pheromone_up - pheromone_down) * a->role[0] * 0.5f;
    
    // Random exploration
    dx += (lcgf(&a->rng) - 0.5f) * 0.02f * a->role[0];
    dy += (lcgf(&a->rng) - 0.5f) * 0.02f * a->role[0];
    
    // Update position
    a->vx = a->vx * 0.8f + dx * 0.2f;
    a->vy = a->vy * 0.8f + dy * 0.2f;
    
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary wrap
    if (a->x < 0) a->x = 0;
    if (a->x > WORLD_SIZE) a->x = WORLD_SIZE;
    if (a->y < 0) a->y = 0;
    if (a->y > WORLD_SIZE) a->y = WORLD_SIZE;
    
    // Resource interaction
    float detect_range = 0.03f + a->role[0] * 0.04f;  // Explore role increases detection
    float grab_range = 0.02f + a->role[1] * 0.02f;    // Collect role increases grab
    
    // Find nearest resource
    int nearest_idx = -1;
    float nearest_dist = 1e6;
    
    for (int r = 0; r < RESOURCES; r++) {
        Resource* res = &resources[r];
        if (res->collected) continue;
        
        float dx = res->x - a->x;
        float dy = res->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range && dist < nearest_dist) {
            nearest_dist = dist;
            nearest_idx = r;
        }
    }
    
    // Collect resource if in range
    if (nearest_idx != -1 && nearest_dist < grab_range) {
        Resource* res = &resources[nearest_idx];
        
        // Collect with bonus from collect role
        float bonus = 1.0f + a->role[1] * 0.5f;
        a->energy += res->value * bonus;
        a->fitness += res->value * bonus;
        
        // Deposit pheromone at resource location (stigmergy)
        deposit_pheromone(pheromones, res->x, res->y, 1.0f + a->role[1], current_tick);
        
        // Mark as collected
        res->collected = 1;
        atomicAdd(resource_counter, 1);
    }
    
    // Communication (broadcast nearest resource location)
    if (nearest_idx != -1 && a->role[2] > 0.3f) {
        Resource* res = &resources[nearest_idx];
        // In real implementation would broadcast to nearby agents
        // Simplified: deposit pheromone at resource location for communication
        deposit_pheromone(pheromones, res->x, res->y, a->role[2] * 0.5f, current_tick);
    }
    
    // Territory defense boost
    float defense_boost = 1.0f;
    int nearby_defenders = 0;
    
    // Simplified: check random other agents
    for (int i = 0; i < 5; i++) {
        int other_idx = (idx + i * 37) % AGENTS;
        if (other_idx == idx) continue;
        
        Agent* other = &agents[other_idx];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.06f && other->arch == a->arch && other->role[3] > 0.3f) {
            nearby_defenders++;
        }
    }
    
    defense_boost += nearby_defenders * 0.2f * a->role[3];
    a->energy *= defense_boost;
    
    // Perturbation resistance from defend role
    if (current_tick % 100 == 0 && lcgf(&a->rng) < 0.1f) {
        float resistance = a->role[3];
        a->energy *= (0.5f + resistance * 0.5f);
    }
}

// Reset collected resources
__global__ void reset_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    resources[idx].collected = 0;
}

int main() {
    printf("Experiment v93: STIGMERY TRAILS\n");
    printf("Testing pheromone trails on specialized vs uniform agents\n");
    printf("Prediction: Specialists benefit more from stigmergy\n\n");
    
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    Pheromone* d_pheromones_spec;
    Pheromone* d_pheromones_uniform;
    int* d_resource_counter;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, PHEROMONE_GRID * PHEROMONE_GRID * sizeof(Pheromone));
    cudaMalloc(&d_pheromones_uniform, PHEROMONE_GRID * PHEROMONE_GRID * sizeof(Pheromone));
    cudaMalloc(&d_resource_counter, sizeof(int));
    
    // Host memory for results
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    float total_fitness_spec = 0.0f;
    float total_fitness_uniform = 0.0f;
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    init_agents<<<grid_spec, block>>>(d_agents_spec, 1);  // Specialized
    init_agents<<<grid_spec, block>>>(d_agents_uniform, 0); // Uniform
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromones<<<grid_ph, block>>>(d_pheromones_spec);
    init_pheromones<<<grid_ph, block>>>(d_pheromones_uniform);
    cudaDeviceSynchronize();
    
    // Run simulation for specialized agents
    printf("Running specialized agents with stigmergy...\n");
    int resources_collected_spec = 0;
    
    for (int tick = 0; tick < TICKS; tick++) {
        int counter = 0;
        cudaMemcpy(d_resource_counter, &counter, sizeof(int), cudaMemcpyHostToDevice);
        
        decay_pheromones<<<grid_ph, block>>>(d_pheromones_spec, tick);
        tick_kernel<<<grid_spec, block>>>(d_agents_spec, d_resources, d_pheromones_spec, 
                                         tick, d_resource_counter);
        
        cudaMemcpy(&counter, d_resource_counter, sizeof(int), cudaMemcpyDeviceToHost);
        resources_collected_spec += counter;
        
        // Respawn resources if many collected
        if (counter > RESOURCES / 2) {
            reset_resources<<<grid_res, block>>>(d_resources);
        }
        
        if (tick % 100 == 0) printf("  Tick %d: collected %d resources\n", tick, counter);
    }
    
    // Copy back specialized results
    cudaMemcpy(h_agents_spec, d_agents_spec, AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
    for (int i = 0; i < AGENTS; i++) {
        total_fitness_spec += h_agents_spec[i].fitness;
    }
    
    // Reinitialize for uniform agents
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromones<<<grid_ph, block>>>(d_pheromones_uniform);
    cudaDeviceSynchronize();
    
    // Run simulation for uniform agents
    printf("\nRunning uniform agents with stigmergy...\n");
    int resources_collected_uniform = 0;
    
    for (int tick = 0; tick < TICKS; tick++) {
        int counter = 0;
        cudaMemcpy(d_resource