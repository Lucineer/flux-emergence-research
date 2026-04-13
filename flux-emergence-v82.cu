/*
CUDA Simulation Experiment v82: Stigmergy with Pheromone Trails
Testing: Whether pheromone trails left at resource locations improve specialist efficiency
Prediction: Pheromones will help uniform agents more than specialists (reducing specialist advantage)
  because specialists already have optimized roles, while uniform agents benefit more from shared information.
Baseline: v8 mechanisms (scarcity, territory, communication) + anti-convergence
Novelty: Agents leave pheromone markers at collected resources that decay over time
  - Pheromone strength = resource value collected
  - Decay rate = 0.95 per tick
  - Detection range = 0.08 (wider than resource detection)
  - Agents are attracted to strongest pheromone in range
  - Specialists should benefit less (already optimized)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RES_COUNT = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Pheromone grid constants
const int GRID_SIZE = 256;  // 256x256 grid
const float CELL_SIZE = 1.0f / GRID_SIZE;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_DETECTION_RANGE = 0.08f;

// Agent archetypes
enum { ARCH_UNIFORM = 0, ARCH_SPECIALIST = 1 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Archetype
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    bool collected;       // Collection status
};

// Pheromone structure for grid
struct PheromoneCell {
    float strength[2];    // Separate pheromone for each archetype
};

// Linear Congruential Generator (device/host)
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize pheromone grid
__global__ void initPheromones(PheromoneCell *grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < GRID_SIZE * GRID_SIZE) {
        grid[idx].strength[0] = 0.0f;
        grid[idx].strength[1] = 0.0f;
    }
}

// Decay pheromones
__global__ void decayPheromones(PheromoneCell *grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < GRID_SIZE * GRID_SIZE) {
        grid[idx].strength[0] *= PHEROMONE_DECAY;
        grid[idx].strength[1] *= PHEROMONE_DECAY;
    }
}

// Initialize agents
__global__ void initAgents(Agent *agents, PheromoneCell *grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    unsigned int seed = idx * 123456789;
    Agent &a = agents[idx];
    
    // Random position
    a.x = lcgf(seed);
    a.y = lcgf(seed);
    a.vx = lcgf(seed) * 0.02f - 0.01f;
    a.vy = lcgf(seed) * 0.02f - 0.01f;
    a.energy = 1.0f;
    a.rng = seed;
    a.fitness = 0.0f;
    
    // Assign archetype (half uniform, half specialist)
    a.arch = (idx < AGENT_COUNT/2) ? ARCH_UNIFORM : ARCH_SPECIALIST;
    
    // Set roles based on archetype
    if (a.arch == ARCH_UNIFORM) {
        // Uniform: all roles equal
        a.role[0] = 0.25f;  // explore
        a.role[1] = 0.25f;  // collect
        a.role[2] = 0.25f;  // communicate
        a.role[3] = 0.25f;  // defend
    } else {
        // Specialist: focus on one role based on hash
        int role_idx = idx % 4;
        for (int i = 0; i < 4; i++) {
            a.role[i] = (i == role_idx) ? 0.7f : 0.1f;
        }
    }
}

// Initialize resources
__global__ void initResources(Resource *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    unsigned int seed = idx * 987654321;
    Resource &r = resources[idx];
    
    // Random position
    r.x = lcgf(seed);
    r.y = lcgf(seed);
    r.value = 0.5f + lcgf(seed) * 0.5f;  // 0.5-1.0
    r.collected = false;
}

// Find grid cell for position
__device__ int getGridCell(float x, float y) {
    int gx = (int)(x * GRID_SIZE) % GRID_SIZE;
    int gy = (int)(y * GRID_SIZE) % GRID_SIZE;
    return gy * GRID_SIZE + gx;
}

// Add pheromone at location
__device__ void addPheromone(PheromoneCell *grid, float x, float y, float strength, int arch) {
    int cell = getGridCell(x, y);
    atomicAdd(&grid[cell].strength[arch], strength);
}

// Get pheromone strength at location
__device__ float getPheromone(PheromoneCell *grid, float x, float y, int arch) {
    int cell = getGridCell(x, y);
    return grid[cell].strength[arch];
}

// Find strongest pheromone in range
__device__ void findStrongestPheromone(PheromoneCell *grid, float x, float y, int arch,
                                      float &dir_x, float &dir_y, float &max_strength) {
    dir_x = 0.0f;
    dir_y = 0.0f;
    max_strength = 0.0f;
    
    int center_cell = getGridCell(x, y);
    int center_x = center_cell % GRID_SIZE;
    int center_y = center_cell / GRID_SIZE;
    
    int range_cells = (int)(PHEROMONE_DETECTION_RANGE / CELL_SIZE) + 1;
    
    for (int dy = -range_cells; dy <= range_cells; dy++) {
        for (int dx = -range_cells; dx <= range_cells; dx++) {
            int gx = (center_x + dx + GRID_SIZE) % GRID_SIZE;
            int gy = (center_y + dy + GRID_SIZE) % GRID_SIZE;
            int cell = gy * GRID_SIZE + gx;
            
            float strength = grid[cell].strength[arch];
            if (strength > max_strength) {
                max_strength = strength;
                dir_x = (gx + 0.5f) * CELL_SIZE - x;
                dir_y = (gy + 0.5f) * CELL_SIZE - y;
            }
        }
    }
}

// Main simulation tick
__global__ void tick(Agent *agents, Resource *resources, PheromoneCell *grid,
                     float *uniform_energy, float *specialist_energy,
                     int *uniform_collect, int *specialist_collect) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = (idx + 1) % AGENT_COUNT;
    Agent &other = agents[other_idx];
    
    if (a.arch == other.arch) {
        float similarity = 0.0f;
        for (int i = 0; i < 4; i++) {
            similarity += fabsf(a.role[i] - other.role[i]);
        }
        similarity = 1.0f - similarity / 4.0f;
        
        if (similarity > 0.9f) {
            // Find dominant role
            int dominant = 0;
            for (int i = 1; i < 4; i++) {
                if (a.role[i] > a.role[dominant]) dominant = i;
            }
            
            // Apply random drift to non-dominant role
            int drift_role;
            do {
                drift_role = (int)(lcgf(a.rng) * 4.0f);
            } while (drift_role == dominant);
            
            a.role[drift_role] += (lcgf(a.rng) - 0.5f) * 0.02f;
            
            // Renormalize
            float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
            for (int i = 0; i < 4; i++) {
                a.role[i] /= sum;
            }
        }
    }
    
    // Pheromone attraction
    float pheromone_dir_x = 0.0f, pheromone_dir_y = 0.0f;
    float pheromone_strength = 0.0f;
    findStrongestPheromone(grid, a.x, a.y, a.arch,
                          pheromone_dir_x, pheromone_dir_y, pheromone_strength);
    
    // Normalize pheromone direction
    float dist = sqrtf(pheromone_dir_x * pheromone_dir_x + 
                      pheromone_dir_y * pheromone_dir_y);
    if (dist > 0.0f && pheromone_strength > 0.1f) {
        pheromone_dir_x /= dist;
        pheromone_dir_y /= dist;
        
        // Weight by explore role
        a.vx += pheromone_dir_x * 0.01f * a.role[0];
        a.vy += pheromone_dir_y * 0.01f * a.role[0];
    }
    
    // Random exploration
    a.vx += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[0];
    a.vy += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[0];
    
    // Velocity damping and position update
    a.vx *= 0.95f;
    a.vy *= 0.95f;
    a.x += a.vx;
    a.y += a.vy;
    
    // Wrap around boundaries
    if (a.x < 0.0f) a.x += 1.0f;
    if (a.x >= 1.0f) a.x -= 1.0f;
    if (a.y < 0.0f) a.y += 1.0f;
    if (a.y >= 1.0f) a.y -= 1.0f;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    // Find nearest resource
    for (int i = 0; i < RES_COUNT; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        // Wrap-around distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx * dx + dy * dy);
        
        // Detection range based on explore role
        float detect_range = 0.03f + a.role[0] * 0.04f;
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    // Collect resource if in range
    if (best_res != -1) {
        Resource &r = resources[best_res];
        float grab_range = 0.02f + a.role[1] * 0.02f;
        
        if (best_dist < grab_range) {
            // Collection bonus based on collect role
            float bonus = 1.0f + a.role[1] * 0.5f;
            a.energy += r.value * bonus;
            a.fitness += r.value * bonus;
            r.collected = true;
            
            // Leave pheromone at collection site
            addPheromone(grid, r.x, r.y, r.value * 2.0f, a.arch);
            
            // Track collections by archetype
            if (a.arch == ARCH_UNIFORM) {
                atomicAdd(uniform_collect, 1);
            } else {
                atomicAdd(specialist_collect, 1);
            }
        }
    }
    
    // Communication (broadcast nearest resource)
    if (a.role[2] > 0.3f && best_res != -1) {
        Resource &r = resources[best_res];
        for (int i = 0; i < AGENT_COUNT; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            
            // Only communicate with same archetype
            if (other.arch != a.arch) continue;
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx > 0.5f) dx -= 1.0f;
            if (dx < -0.5f) dx += 1.0f;
            if (dy > 0.5f) dy -= 1.0f;
            if (dy < -0.5f) dy += 1.0f;
            
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 0.06f) {
                // Attract other agent toward resource
                float res_dx = r.x - other.x;
                float res_dy = r.y - other.y;
                if (res_dx > 0.5f) res_dx -= 1.0f;
                if (res_dx < -0.5f) res_dx += 1.0f;
                if (res_dy > 0.5f) res_dy -= 1.0f;
                if (res_dy < -0.5f) res_dy += 1.0f;
                
                float res_dist = sqrtf(res_dx * res_dx + res_dy * res_dy);
                if (res_dist > 0.0f) {
                    other.vx += (res_dx / res_dist) * 0.02f * a.role[2];
                    other.vy += (res_dy / res_dist) * 0.02f * a.role[2];
                }
            }
        }
    }
    
    // Territory defense
    if (a.role[3] > 0.3f) {
        int nearby_defenders = 0;
        for (int i = 0; i < AGENT_COUNT; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            if (other.arch != a.arch) continue;
            if (other.role[3] < 0.3f) continue;
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx > 0.5f) dx -= 1.0f;
            if (dx < -0.5f) dx += 1.0f;
            if (dy > 0.5f) dy -= 1.0f;
            if (dy < -0.5f) dy += 1.0f;
            
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 0.1f) {
                nearby_defenders++;
            }
        }
        
        // Defense bonus
        float defense_bonus = 1.0f + nearby_defenders * 0.2f;
        a.energy *= defense_bonus;
    }
    
    // Perturbation (energy halving) - defenders resist
    if (lcgf(a.rng) < 0.01f) {
        float resistance = a.role[3] > 0.3f ? 0.5f : 1.0f;
        a.energy *= (1.0f - 0.5f * resistance);
    }
    
    // Track energy by archetype
    if (a.arch == ARCH_UNIFORM) {
        atomicAdd(uniform_energy, a.energy);
    } else {
        atomicAdd