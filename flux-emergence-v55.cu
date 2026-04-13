// CUDA Simulation Experiment v55: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist efficiency
// Prediction: Pheromones will help uniform agents more than specialists (falsifying advantage)
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Novel: Agents leave pheromone trails at collected resource sites, others follow gradients

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RES_COUNT = 128;
const int TICKS = 500;
const int ARCH_COUNT = 4;
const int PHEROMONE_GRID = 256; // 256x256 grid
const float WORLD_SIZE = 1.0f;
const float CELL_SIZE = WORLD_SIZE / PHEROMONE_GRID;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Resource
struct Resource {
    float x, y;
    float value;
    bool collected;
    int last_collected_tick;
};

// Pheromone grid cell
struct PheromoneCell {
    float strength[ARCH_COUNT];
    int last_update;
};

// Agent
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCH_COUNT]; // 4 behavioral dimensions
    float fitness;
    int arch; // 0-3 archetype
    unsigned int rng;
    
    // Memory for pheromone following
    float target_x, target_y;
    int memory_timer;
};

// Simulation state in device memory
__device__ Agent agents[AGENT_COUNT];
__device__ Resource resources[RES_COUNT];
__device__ PheromoneCell pheromone_grid[PHEROMONE_GRID][PHEROMONE_GRID];
__device__ int current_tick = 0;

// Utility kernels
__global__ void initPheromoneGrid() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    int i = idx / PHEROMONE_GRID;
    int j = idx % PHEROMONE_GRID;
    
    for (int a = 0; a < ARCH_COUNT; a++) {
        pheromone_grid[i][j].strength[a] = 0.0f;
    }
    pheromone_grid[i][j].last_update = -1000;
}
__global__ void initAgents() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    a.rng = 123456789 + idx * 987654321;
    
    // Position
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = (lcgf(a.rng) - 0.5f) * 0.02f;
    a.vy = (lcgf(a.rng) - 0.5f) * 0.02f;
    
    // Energy
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.memory_timer = 0;
    a.target_x = a.x;
    a.target_y = a.y;
    
    // Assign archetype
    a.arch = idx % ARCH_COUNT;
    
    // SPECIALIZED GROUP (first half): strong in their archetype role
    if (idx < AGENT_COUNT / 2) {
        for (int i = 0; i < ARCH_COUNT; i++) {
            a.role[i] = 0.1f; // Baseline
        }
        a.role[a.arch] = 0.7f; // Strong specialization
        
        // Anti-convergence: add small random variation
        for (int i = 0; i < ARCH_COUNT; i++) {
            if (i != a.arch) {
                a.role[i] += (lcgf(a.rng) - 0.5f) * 0.02f;
            }
        }
    }
    // UNIFORM CONTROL GROUP (second half): all roles equal
    else {
        for (int i = 0; i < ARCH_COUNT; i++) {
            a.role[i] = 0.25f;
        }
        // Small variation to prevent perfect symmetry
        for (int i = 0; i < ARCH_COUNT; i++) {
            a.role[i] += (lcgf(a.rng) - 0.5f) * 0.01f;
        }
    }
    
    // Normalize
    float sum = 0.0f;
    for (int i = 0; i < ARCH_COUNT; i++) sum += a.role[i];
    for (int i = 0; i < ARCH_COUNT; i++) a.role[i] /= sum;
}

__global__ void initResources() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    Resource &r = resources[idx];
    
    // Use thread-local RNG for resource initialization
    unsigned int rng = 123456789 + idx * 7654321;
    
    // Power-law distribution (clustered resources)
    float angle = lcgf(rng) * 6.28318530718f;
    float radius = pow(lcgf(rng), 1.5f) * 0.4f;
    r.x = 0.5f + cos(angle) * radius;
    r.y = 0.5f + sin(angle) * radius;
    
    r.value = 0.5f + lcgf(rng) * 0.5f;
    r.collected = false;
    r.last_collected_tick = -100;
}

// Pheromone evaporation and diffusion
__global__ void updatePheromones() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= PHEROMONE_GRID || j >= PHEROMONE_GRID) return;
    
    PheromoneCell &cell = pheromone_grid[i][j];
    
    // Evaporation
    for (int a = 0; a < ARCH_COUNT; a++) {
        cell.strength[a] *= 0.95f; // 5% evaporation per tick
    }
    
    // Simple diffusion to neighbors
    if (i > 0 && j > 0 && i < PHEROMONE_GRID-1 && j < PHEROMONE_GRID-1) {
        float diffusion_rate = 0.1f;
        for (int a = 0; a < ARCH_COUNT; a++) {
            float avg = (pheromone_grid[i-1][j].strength[a] +
                        pheromone_grid[i+1][j].strength[a] +
                        pheromone_grid[i][j-1].strength[a] +
                        pheromone_grid[i][j+1].strength[a]) * 0.25f;
            cell.strength[a] = cell.strength[a] * (1.0f - diffusion_rate) + avg * diffusion_rate;
        }
    }
    
    // Age out old pheromones
    if (current_tick - cell.last_update > 100) {
        for (int a = 0; a < ARCH_COUNT; a++) {
            cell.strength[a] *= 0.8f;
        }
    }
}

// Main simulation kernel
__global__ void tick() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    current_tick = current_tick; // Ensure available
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: detect similarity with same-arch agents
    if (idx % 32 == 0) { // Check periodically
        int similar_count = 0;
        int total_count = 0;
        
        for (int i = 0; i < min(16, AGENT_COUNT); i++) {
            int other_idx = (idx + i * 31) % AGENT_COUNT;
            if (agents[other_idx].arch == a.arch) {
                total_count++;
                float similarity = 0.0f;
                for (int r = 0; r < ARCH_COUNT; r++) {
                    similarity += 1.0f - fabsf(a.role[r] - agents[other_idx].role[r]);
                }
                similarity /= ARCH_COUNT;
                if (similarity > 0.9f) similar_count++;
            }
        }
        
        if (total_count > 3 && similar_count * 2 > total_count) {
            // Random drift on non-dominant role
            int drift_role = (a.arch + 1 + (a.rng % (ARCH_COUNT - 1))) % ARCH_COUNT;
            a.role[drift_role] += (lcgf(a.rng) - 0.5f) * 0.01f;
            
            // Renormalize
            float sum = 0.0f;
            for (int r = 0; r < ARCH_COUNT; r++) sum += a.role[r];
            for (int r = 0; r < ARCH_COUNT; r++) a.role[r] /= sum;
        }
    }
    
    // Role-based behavior parameters
    float detect_range = 0.03f + a.role[0] * 0.04f; // Explorer role
    float grab_range = 0.02f + a.role[1] * 0.02f;   // Collector role
    float comm_range = 0.04f + a.role[2] * 0.02f;   // Communicator role
    float defend_bonus = a.role[3];                 // Defender role
    
    // Check pheromone trail if memory timer expired
    if (a.memory_timer <= 0) {
        int grid_x = min(PHEROMONE_GRID-1, max(0, (int)(a.x / WORLD_SIZE * PHEROMONE_GRID)));
        int grid_y = min(PHEROMONE_GRID-1, max(0, (int)(a.y / WORLD_SIZE * PHEROMONE_GRID)));
        
        float best_strength = -1.0f;
        int best_dx = 0, best_dy = 0;
        
        // Look at 3x3 neighborhood for strongest pheromone of our archetype
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = grid_x + dx;
                int ny = grid_y + dy;
                if (nx >= 0 && nx < PHEROMONE_GRID && ny >= 0 && ny < PHEROMONE_GRID) {
                    float strength = pheromone_grid[nx][ny].strength[a.arch];
                    if (strength > best_strength) {
                        best_strength = strength;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
        }
        
        // If found strong enough pheromone, follow gradient
        if (best_strength > 0.1f) {
            a.target_x = a.x + best_dx * CELL_SIZE * 3.0f;
            a.target_y = a.y + best_dy * CELL_SIZE * 3.0f;
            a.memory_timer = 10 + (a.rng % 20); // Follow for 10-30 ticks
        } else {
            // Random walk
            a.target_x = a.x + (lcgf(a.rng) - 0.5f) * 0.1f;
            a.target_y = a.y + (lcgf(a.rng) - 0.5f) * 0.1f;
            a.memory_timer = 5 + (a.rng % 15);
        }
    }
    
    a.memory_timer--;
    
    // Move toward target with some randomness
    float dx = a.target_x - a.x;
    float dy = a.target_y - a.y;
    float dist = sqrtf(dx*dx + dy*dy + 1e-6f);
    
    a.vx = a.vx * 0.8f + (dx / dist) * 0.02f + (lcgf(a.rng) - 0.5f) * 0.005f;
    a.vy = a.vy * 0.8f + (dy / dist) * 0.02f + (lcgf(a.rng) - 0.5f) * 0.005f;
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // Wrap around world
    if (a.x < 0) a.x = 0, a.vx = fabsf(a.vx);
    if (a.x > WORLD_SIZE) a.x = WORLD_SIZE, a.vx = -fabsf(a.vx);
    if (a.y < 0) a.y = 0, a.vy = fabsf(a.vy);
    if (a.y > WORLD_SIZE) a.y = WORLD_SIZE, a.vy = -fabsf(a.vy);
    
    // Resource collection
    float best_dist = 1e6f;
    int best_res = -1;
    
    for (int i = 0; i < RES_COUNT; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist = dx*dx + dy*dy;
        
        if (dist < detect_range * detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    // Collect if in grab range
    if (best_res != -1 && best_dist < grab_range * grab_range) {
        Resource &r = resources[best_res];
        
        // Territory bonus: nearby defenders of same archetype
        float territory_bonus = 1.0f;
        int defender_count = 0;
        for (int i = 0; i < AGENT_COUNT; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx*dx + dy*dy < 0.04f * 0.04f && other.arch == a.arch) {
                defender_count++;
            }
        }
        territory_bonus += defender_count * 0.2f * defend_bonus;
        
        // Collector role bonus
        float collector_bonus = 1.0f + a.role[1] * 0.5f;
        
        // Gain energy
        float gain = r.value * territory_bonus * collector_bonus;
        a.energy += gain;
        a.fitness += gain;
        
        // Mark resource collected
        r.collected = true;
        r.last_collected_tick = current_tick;
        
        // NOVEL MECHANISM: Leave pheromone at collection site
        int grid_x = min(PHEROMONE_GRID-1, max(0, (int)(r.x / WORLD_SIZE * PHEROMONE_GRID)));
        int grid_y = min(PHEROMONE_GRID-1, max(0, (int)(r.y / WORLD_SIZE * PHEROMONE_GRID)));
        
        atomicAdd(&pheromone_grid[grid_x][grid_y].strength[a.arch], 1.0f);
        pheromone_grid[grid_x][grid_y].last_update = current_tick;
        
        // Communicate location to nearby agents (communicator role)
        if (a.role[2] > 0.3f) {
            for (int i = 0; i < AGENT_COUNT; i++) {
                if (i == idx) continue;
                Agent &other = agents[i];
                float dx = other.x - a.x;
                float dy = other.y - a.y;
                if (dx*dx + dy*dy < comm_range * comm_range) {
                    // Share resource location
                    other.target_x = r.x;
                    other.target_y = r.y;
                    other.memory_timer = 15;
                }
            }
        }
    }
    
    // Resource respawn (every 50 ticks)
    if (current_tick % 50 == 0) {
        for (int i = 0; i < RES_COUNT; i++) {
            if (resources[i].collected && current_tick - resources[i].last_collected_tick > 10) {
                resources[i].collected = false;
                // Slight position jitter
                resources[i].x += (lcgf(a.rng) - 0.5f) * 0.05f;
                resources[i].y += (lcgf(a.rng) - 0.5f) * 0.05f;
                resources[i].x = fmaxf(0.0f, fminf(WORLD_SIZE, resources[i].x));
                resources[i].y = fmaxf(0.0f, fminf(WORLD_SIZE, resources[i].y));
            }
        }
    }
    
    // Perturbation