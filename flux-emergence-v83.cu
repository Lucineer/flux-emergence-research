
/*
CUDA Simulation Experiment v83: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination beyond baseline v8,
            especially for explorers who can follow trails to recent finds.
            Expect specialization metric >0.794 and efficiency >1.61x.
Baseline: v8 mechanisms (scarcity, territory, comms) included.
Novelty: Pheromone grid with diffusion and evaporation.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int GRID_SIZE = 256;          // Pheromone grid resolution
const float WORLD_SIZE = 1.0f;
const float PHEROMONE_DECAY = 0.95f; // Per tick
const float PHEROMONE_DIFFUSE = 0.1f; // Diffusion rate
const float PHEROMONE_STRENGTH = 0.5f; // Initial strength when deposited

// Agent archetypes
enum { ARCH_GENERALIST = 0, ARCH_SPECIALIST = 1 };

// Agent struct
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // [explore, collect, communicate, defend]
    float fitness;        // Accumulated resource value
    int arch;             // Archetype
    unsigned int rng;     // RNG state
};

// Resource struct
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // 0=available, 1=collected
};

// Pheromone grid (global memory)
float* d_pheromone;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int& state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int arch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    unsigned int seed = idx * 123456789 + arch * 987654321;
    agents[idx].x = lcgf(seed) * WORLD_SIZE;
    agents[idx].y = lcgf(seed) * WORLD_SIZE;
    agents[idx].vx = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].arch = arch;
    agents[idx].rng = seed;
    
    // Role initialization based on archetype
    if (arch == ARCH_SPECIALIST) {
        // Specialists: strong in one role (0.7), weak in others (0.1)
        int specialty = idx % 4;
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = (i == specialty) ? 0.7f : 0.1f;
        }
    } else {
        // Generalists: uniform roles
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int seed = idx * 135791113;
    resources[idx].x = lcgf(seed) * WORLD_SIZE;
    resources[idx].y = lcgf(seed) * WORLD_SIZE;
    resources[idx].value = 0.5f + lcgf(seed) * 0.5f; // 0.5-1.0
    resources[idx].collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromone(float* pheromone) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE * GRID_SIZE) return;
    pheromone[idx] = 0.0f;
}

// Diffuse pheromone kernel
__global__ void diffuse_pheromone(float* pheromone) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) return;
    
    int idx = y * GRID_SIZE + x;
    float center = pheromone[idx];
    
    // Simple diffusion to neighbors (5-point stencil)
    float diffuse_sum = 0.0f;
    int count = 0;
    
    if (x > 0) { diffuse_sum += pheromone[idx - 1]; count++; }
    if (x < GRID_SIZE - 1) { diffuse_sum += pheromone[idx + 1]; count++; }
    if (y > 0) { diffuse_sum += pheromone[idx - GRID_SIZE]; count++; }
    if (y < GRID_SIZE - 1) { diffuse_sum += pheromone[idx + GRID_SIZE]; count++; }
    
    if (count > 0) {
        pheromone[idx] = center * (1.0f - PHEROMONE_DIFFUSE) + 
                        (diffuse_sum / count) * PHEROMONE_DIFFUSE;
    }
    
    // Decay
    pheromone[idx] *= PHEROMONE_DECAY;
}

// Deposit pheromone at resource location
__device__ void deposit_pheromone(float x, float y, float* pheromone) {
    int grid_x = (int)((x / WORLD_SIZE) * GRID_SIZE);
    int grid_y = (int)((y / WORLD_SIZE) * GRID_SIZE);
    grid_x = max(0, min(GRID_SIZE - 1, grid_x));
    grid_y = max(0, min(GRID_SIZE - 1, grid_y));
    
    int idx = grid_y * GRID_SIZE + grid_x;
    atomicAdd(&pheromone[idx], PHEROMONE_STRENGTH);
}

// Sample pheromone at position
__device__ float sample_pheromone(float x, float y, float* pheromone) {
    int grid_x = (int)((x / WORLD_SIZE) * GRID_SIZE);
    int grid_y = (int)((y / WORLD_SIZE) * GRID_SIZE);
    grid_x = max(0, min(GRID_SIZE - 1, grid_x));
    grid_y = max(0, min(GRID_SIZE - 1, grid_y));
    
    return pheromone[grid_y * GRID_SIZE + grid_x];
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, float* pheromone, 
                     int tick_num, int* resource_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: detect similarity > 0.9, apply drift
    float role_sum = 0.0f;
    for (int i = 0; i < 4; i++) role_sum += a.role[i] * a.role[i];
    if (role_sum > 0.9f) {
        int drift_role = lcgf(a.rng) * 4;
        a.role[drift_role] += (lcgf(a.rng) * 0.02f - 0.01f);
        a.role[drift_role] = max(0.0f, min(1.0f, a.role[drift_role]));
    }
    
    // Pheromone influence on exploration
    float pheromone_level = sample_pheromone(a.x, a.y, pheromone);
    float explore_bias = a.role[0] * pheromone_level * 0.1f;
    
    // Movement with pheromone bias
    a.vx += (lcgf(a.rng) * 0.004f - 0.002f) + explore_bias;
    a.vy += (lcgf(a.rng) * 0.004f - 0.002f) + explore_bias;
    
    // Velocity damping
    a.vx *= 0.98f;
    a.vy *= 0.98f;
    
    // Update position (toroidal world)
    a.x += a.vx;
    a.y += a.vy;
    if (a.x < 0) a.x += WORLD_SIZE;
    if (a.x >= WORLD_SIZE) a.x -= WORLD_SIZE;
    if (a.y < 0) a.y += WORLD_SIZE;
    if (a.y >= WORLD_SIZE) a.y -= WORLD_SIZE;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    // Explorer detection range
    float detect_range = 0.03f + a.role[0] * 0.04f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource& r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        // Toroidal distance
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        
        float dist = sqrtf(dx * dx + dy * dy);
        
        if (dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
        
        // Collection
        float grab_range = 0.02f + a.role[1] * 0.02f;
        if (dist < grab_range) {
            float value = r.value;
            // Collector bonus
            if (a.role[1] > 0.3f) value *= 1.5f;
            
            // Defender territory bonus
            int nearby_defenders = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent& other = agents[j];
                if (other.arch != a.arch) continue;
                
                float odx = other.x - a.x;
                float ody = other.y - a.y;
                if (odx > 0.5f * WORLD_SIZE) odx -= WORLD_SIZE;
                if (odx < -0.5f * WORLD_SIZE) odx += WORLD_SIZE;
                if (ody > 0.5f * WORLD_SIZE) ody -= WORLD_SIZE;
                if (ody < -0.5f * WORLD_SIZE) ody += WORLD_SIZE;
                
                if (sqrtf(odx * odx + ody * ody) < 0.05f && other.role[3] > 0.3f) {
                    nearby_defenders++;
                }
            }
            value *= (1.0f + nearby_defenders * 0.2f);
            
            a.fitness += value;
            a.energy += value * 0.1f;
            r.collected = 1;
            atomicAdd(resource_counter, 1);
            
            // Deposit pheromone at collected resource location
            deposit_pheromone(r.x, r.y, pheromone);
            break;
        }
    }
    
    // Communication
    if (a.role[2] > 0.3f && best_res != -1 && best_dist < detect_range) {
        Resource& r = resources[best_res];
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent& other = agents[j];
            if (other.arch != a.arch) continue;
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
            if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
            if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
            if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
            
            if (sqrtf(dx * dx + dy * dy) < 0.06f) {
                // Attract neighbor toward resource
                float rx = r.x - other.x;
                float ry = r.y - other.y;
                if (rx > 0.5f * WORLD_SIZE) rx -= WORLD_SIZE;
                if (rx < -0.5f * WORLD_SIZE) rx += WORLD_SIZE;
                if (ry > 0.5f * WORLD_SIZE) ry -= WORLD_SIZE;
                if (ry < -0.5f * WORLD_SIZE) ry += WORLD_SIZE;
                
                float len = sqrtf(rx * rx + ry * ry);
                if (len > 0.001f) {
                    other.vx += (rx / len) * 0.01f * a.role[2];
                    other.vy += (ry / len) * 0.01f * a.role[2];
                }
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0) {
        // Defenders resist perturbation
        if (a.role[3] < 0.5f || lcgf(a.rng) > a.role[3]) {
            a.energy *= 0.5f;
            a.vx += lcgf(a.rng) * 0.1f - 0.05f;
            a.vy += lcgf(a.rng) * 0.1f - 0.05f;
        }
    }
    
    // Coupling with same archetype
    for (int j = idx + 1; j < min(idx + 10, AGENTS); j++) {
        Agent& other = agents[j];
        if (other.arch != a.arch) continue;
        
        float dx = other.x - a.x;
        float dy = other.y - a.y;
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist < 0.02f) {
            // Role coupling
            for (int k = 0; k < 4; k++) {
                float diff = other.role[k] - a.role[k];
                a.role[k] += diff * 0.02f;
                other.role[k] -= diff * 0.02f;
            }
        }
    }
}

int main() {
    // Allocate host memory
    Agent* h_agents = new Agent[AGENTS];
    Resource* h_resources = new Resource[RESOURCES];
    int* h_resource_counter = new int[2]; // [specialist, generalist]
    
    // Allocate device memory
    Agent* d_agents_spec;
    Agent* d_agents_gen;
    Resource* d_resources;
    int* d_resource_counter;
    
    cudaMalloc(&d_agents_spec, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_agents_gen, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    cudaMalloc(&d_resource_counter, sizeof(int) * 2);
    cudaMalloc(&d_pheromone, sizeof(float) * GRID_SIZE * GRID_SIZE);
    
    // Initialize
    dim3 block(256);
    dim3 grid_agents((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((GRID_SIZE * GRID_SIZE + 255) / 256);
    
    init_agents<<<grid_agents, block>>>(d_agents_spec, ARCH_SPECIALIST);
    init_agents<<<grid_agents, block>>>(d_agents_gen, ARCH_GENERALIST);
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromone<<<grid_ph, block>>>(d_pheromone);
    
    cudaDeviceSynchronize();
    
    // Simulation loop
    float total_fitness_spec = 0.0f;
    float total_fitness_gen = 0.0f;
    
    for (int tick = 0; tick < TICKS; tick++) {
        // Reset resource counters
        h_resource_counter[0] = 0;
        h_resource_counter[1] = 0;
        cudaMemcpy(d_resource_counter, h_resource_counter, sizeof(int) * 2, cudaMemcpyHostToDevice);
        
        // Reset resources every 50 ticks (scarcity)
        if (tick % 50 == 0) {
            init_resources<<<grid_res