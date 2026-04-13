
/*
CUDA Simulation Experiment v77: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at collected resource locations that decay over time.
Prediction: Pheromones will enhance specialist efficiency by 20-30% over baseline v8,
            as they create persistent environmental memory that guides exploration.
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence) included.
Comparison: Specialized archetypes (role[arch]=0.7) vs uniform control (all roles=0.25).
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RES_COUNT = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const int PHEROMONE_GRID = 256; // 256x256 grid
const int PHEROMONE_COUNT = PHEROMONE_GRID * PHEROMONE_GRID;
const float WORLD_SIZE = 1.0f;
const float CELL_SIZE = WORLD_SIZE / PHEROMONE_GRID;

// Agent structure
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4]; // explore, collect, communicate, defend
    float fitness;
    int arch;
    unsigned int rng;
};

// Resource structure
struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone structure
struct Pheromone {
    float strength[4]; // one per archetype
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
__global__ void init_agents(Agent *agents, int arch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    unsigned int seed = idx * 17 + arch * 7919;
    agents[idx].x = lcgf(seed) * WORLD_SIZE;
    agents[idx].y = lcgf(seed) * WORLD_SIZE;
    agents[idx].vx = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].arch = arch;
    agents[idx].rng = seed * (idx + 1);
    
    // Specialized archetypes vs uniform control
    if (arch == 0) { // Uniform control group
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.25f;
        }
    } else { // Specialized archetypes
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.1f;
        }
        agents[idx].role[arch - 1] = 0.7f; // Primary role based on archetype
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    unsigned int seed = idx * 137;
    resources[idx].x = lcgf(seed) * WORLD_SIZE;
    resources[idx].y = lcgf(seed) * WORLD_SIZE;
    resources[idx].value = 0.8f + lcgf(seed) * 0.4f;
    resources[idx].collected = 0;
}

// Initialize pheromones kernel
__global__ void init_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_COUNT) return;
    
    for (int i = 0; i < 4; i++) {
        pheromones[idx].strength[i] = 0.0f;
    }
}

// Decay pheromones kernel
__global__ void decay_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_COUNT) return;
    
    for (int i = 0; i < 4; i++) {
        pheromones[idx].strength[i] *= 0.95f; // 5% decay per tick
    }
}

// Get pheromone grid index
__device__ int get_pheromone_index(float x, float y) {
    int gx = (int)(x / WORLD_SIZE * PHEROMONE_GRID) % PHEROMONE_GRID;
    int gy = (int)(y / WORLD_SIZE * PHEROMONE_GRID) % PHEROMONE_GRID;
    return gy * PHEROMONE_GRID + gx;
}

// Add pheromone kernel
__global__ void add_pheromone(Pheromone *pheromones, float x, float y, int arch, float strength) {
    int idx = get_pheromone_index(x, y);
    atomicAdd(&pheromones[idx].strength[arch], strength);
}

// Main simulation kernel
__global__ void tick_kernel(Agent *agents, Resource *resources, Pheromone *pheromones, 
                           int tick, int *resource_collected) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with nearby agents
    int similar_count = 0;
    float role_sim_threshold = 0.9f;
    
    // Movement with pheromone guidance
    float dx = 0.0f, dy = 0.0f;
    
    // Sample nearby pheromones (3x3 grid around current position)
    float pheromone_strength[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int center_idx = get_pheromone_index(a.x, a.y);
    
    for (int dyi = -1; dyi <= 1; dyi++) {
        for (int dxi = -1; dxi <= 1; dxi++) {
            int grid_x = (center_idx % PHEROMONE_GRID + dxi + PHEROMONE_GRID) % PHEROMONE_GRID;
            int grid_y = (center_idx / PHEROMONE_GRID + dyi + PHEROMONE_GRID) % PHEROMONE_GRID;
            int neighbor_idx = grid_y * PHEROMONE_GRID + grid_x;
            
            for (int arch = 0; arch < 4; arch++) {
                pheromone_strength[arch] += pheromones[neighbor_idx].strength[arch];
            }
        }
    }
    
    // Pheromone-based movement: move toward strongest pheromone of same archetype
    float max_strength = 0.0f;
    int strongest_dir = -1;
    
    // Check 8 directions for pheromone gradient
    for (int dir = 0; dir < 8; dir++) {
        float angle = dir * 3.14159f / 4.0f;
        float sample_x = a.x + cosf(angle) * CELL_SIZE * 3.0f;
        float sample_y = a.y + sinf(angle) * CELL_SIZE * 3.0f;
        int sample_idx = get_pheromone_index(sample_x, sample_y);
        float strength = pheromones[sample_idx].strength[a.arch];
        
        if (strength > max_strength) {
            max_strength = strength;
            strongest_dir = dir;
        }
    }
    
    if (max_strength > 0.1f && strongest_dir != -1) {
        // Follow pheromone gradient
        float angle = strongest_dir * 3.14159f / 4.0f;
        dx += cosf(angle) * 0.01f * a.role[0]; // Explore role amplifies
        dy += sinf(angle) * 0.01f * a.role[0];
    } else {
        // Random exploration
        dx = lcgf(a.rng) * 0.02f - 0.01f;
        dy = lcgf(a.rng) * 0.02f - 0.01f;
    }
    
    // Communication role: broadcast resource locations
    float nearest_res_x = 0.0f, nearest_res_y = 0.0f;
    float nearest_dist = 1.0f;
    
    // Find nearest resource
    for (int i = 0; i < RES_COUNT; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dist = hypotf(a.x - r.x, a.y - r.y);
        if (dist < nearest_dist) {
            nearest_dist = dist;
            nearest_res_x = r.x;
            nearest_res_y = r.y;
        }
    }
    
    // If communicator and found resource, influence movement
    if (a.role[2] > 0.3f && nearest_dist < 0.06f) {
        dx += (nearest_res_x - a.x) * 0.005f * a.role[2];
        dy += (nearest_res_y - a.y) * 0.005f * a.role[2];
    }
    
    // Update velocity and position
    a.vx = a.vx * 0.8f + dx * 0.2f;
    a.vy = a.vy * 0.8f + dy * 0.2f;
    
    float speed = hypotf(a.vx, a.vy);
    if (speed > 0.03f) {
        a.vx *= 0.03f / speed;
        a.vy *= 0.03f / speed;
    }
    
    a.x += a.vx;
    a.y += a.vy;
    
    // World wrap
    if (a.x < 0) a.x += WORLD_SIZE;
    if (a.x >= WORLD_SIZE) a.x -= WORLD_SIZE;
    if (a.y < 0) a.y += WORLD_SIZE;
    if (a.y >= WORLD_SIZE) a.y -= WORLD_SIZE;
    
    // Resource collection
    for (int i = 0; i < RES_COUNT; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dist = hypotf(a.x - r.x, a.y - r.y);
        float grab_range = 0.02f + a.role[1] * 0.02f; // Collect role increases range
        
        if (dist < grab_range) {
            float bonus = 1.0f + a.role[1] * 0.5f; // 50% bonus for collectors
            a.energy += r.value * bonus;
            a.fitness += r.value * bonus;
            r.collected = 1;
            atomicAdd(resource_collected, 1);
            
            // Leave pheromone at collected resource location
            float pheromone_str = 1.0f + a.role[3] * 2.0f; // Defenders leave stronger marks
            add_pheromone<<<1, 1>>>(pheromones, r.x, r.y, a.arch, pheromone_str);
            
            break;
        }
    }
    
    // Territory defense bonus
    int defenders_nearby = 0;
    for (int j = 0; j < AGENT_COUNT; j++) {
        if (j == idx) continue;
        Agent &other = agents[j];
        
        float dist = hypotf(a.x - other.x, a.y - other.y);
        if (dist < 0.04f) {
            // Role similarity check
            float similarity = 0.0f;
            for (int k = 0; k < 4; k++) {
                similarity += fminf(a.role[k], other.role[k]);
            }
            if (similarity > role_sim_threshold) {
                similar_count++;
            }
            
            // Defender bonus
            if (dist < 0.03f && a.arch == other.arch && other.role[3] > 0.5f) {
                defenders_nearby++;
            }
        }
    }
    
    // Defense bonus: 20% per nearby defender
    if (a.role[3] > 0.3f && defenders_nearby > 0) {
        a.energy *= 1.0f + defenders_nearby * 0.2f;
    }
    
    // Anti-convergence drift
    if (similar_count > 2) {
        // Find dominant role
        int dominant = 0;
        for (int k = 1; k < 4; k++) {
            if (a.role[k] > a.role[dominant]) dominant = k;
        }
        
        // Apply drift to non-dominant roles
        for (int k = 0; k < 4; k++) {
            if (k != dominant) {
                a.role[k] += lcgf(a.rng) * 0.02f - 0.01f;
                a.role[k] = fmaxf(0.1f, fminf(0.8f, a.role[k]));
            }
        }
        
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int k = 0; k < 4; k++) {
            a.role[k] /= sum;
        }
    }
    
    // Perturbation every 50 ticks
    if (tick % 50 == 0) {
        // Defenders resist perturbation
        float resistance = a.role[3] * 0.5f;
        if (lcgf(a.rng) > resistance) {
            a.energy *= 0.5f;
            a.vx += lcgf(a.rng) * 0.1f - 0.05f;
            a.vy += lcgf(a.rng) * 0.1f - 0.05f;
        }
    }
}

int main() {
    // Allocate memory
    Agent *agents;
    Resource *resources;
    Pheromone *pheromones;
    int *resource_collected;
    
    cudaMallocManaged(&agents, sizeof(Agent) * AGENT_COUNT);
    cudaMallocManaged(&resources, sizeof(Resource) * RES_COUNT);
    cudaMallocManaged(&pheromones, sizeof(Pheromone) * PHEROMONE_COUNT);
    cudaMallocManaged(&resource_collected, sizeof(int));
    
    // Initialize
    *resource_collected = 0;
    
    // Create specialized archetypes (256 each) + uniform control (256)
    dim3 block(256);
    dim3 grid_agents((AGENT_COUNT + 255) / 256);
    dim3 grid_res((RES_COUNT + 255) / 256);
    dim3 grid_pheromone((PHEROMONE_COUNT + 255) / 256);
    
    // Initialize uniform control group (arch=0)
    init_agents<<<grid_agents, block>>>(agents, 0);
    
    // Initialize specialized archetypes (arch=1-4)
    for (int arch = 1; arch <= 4; arch++) {
        init_agents<<<grid_agents, block>>>(agents + (arch-1)*256, arch);
    }
    
    init_resources<<<grid_res, block>>>(resources);
    init_pheromones<<<grid_pheromone, block>>>(pheromones);
    
    cudaDeviceSynchronize();
    
    // Track fitness per archetype
    float fitness_per_arch[5] = {0}; // 0=uniform, 1-4=specialized
    
    // Main simulation loop
    for (int tick = 0; tick < TICKS; tick++) {
        // Decay pheromones
        decay_pheromones<<<grid_pheromone, block>>>(pheromones);
        
        // Run simulation kernel
        tick_kernel<<<grid_agents, block>>>(agents, resources, pheromones, tick, resource_collected);
        
        cudaDeviceSynchronize();
        
        // Respawn resources if all collected
        if (*resource_collected >= RES_COUNT) {
            init_resources<<<grid_res, block>>>(resources);
            *resource_collected = 0;
            cudaDeviceSynchronize();
        }
        
        // Collect fitness data every 100 ticks
        if (tick % 100 == 99) {
            for (int i = 0; i < AGENT_COUNT; i++) {
                int arch_idx = (i < 256) ? 0 : ((i / 256) + 1);
                if (arch_idx > 4) arch_idx = 4;
                fitness_per_arch[arch_idx] += agents[i].fitness;
                agents[i].fitness = 0; // Reset for next period
            }
        }
    }
    
    // Calculate final results
    float uniform_fitness = 0.0f;
    float specialized_fitness = 0.0