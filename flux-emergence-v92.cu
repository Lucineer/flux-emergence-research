
/*
CUDA Simulation Experiment v92
Testing: Stigmergy with pheromone trails at resource locations
Prediction: Pheromone trails will improve specialist efficiency by 20-30% over baseline v8
    because specialists can follow trails left by others of same archetype
Novel mechanism: Agents deposit pheromone when collecting resources, pheromone decays over time,
    agents sense and move toward strongest pheromone of their archetype
Baseline: Includes all v8 confirmed mechanisms (scarcity, territory, comms)
Comparison: Specialized agents (role[arch]=0.7) vs uniform control (all roles=0.25)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const int PHEROMONE_GRID = 256; // 256x256 grid for pheromone tracking
const float WORLD_SIZE = 1.0f;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Resource struct
struct Resource {
    float x, y;
    float value;
    bool collected;
    int last_collected_tick;
};

// Pheromone struct for grid-based tracking
struct PheromoneGrid {
    float strength[ARCHETYPES];
    int last_updated;
};

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPES]; // behavioral tendencies
    float fitness;
    int arch; // archetype 0-3
    unsigned int rng;
    
    // For pheromone following
    float target_x, target_y;
    int pheromone_memory;
};

// Global device arrays
__device__ Agent d_agents[AGENTS];
__device__ Resource d_resources[RESOURCES];
__device__ PheromoneGrid d_pheromone[PHEROMONE_GRID][PHEROMONE_GRID];
__device__ int d_tick_counter = 0;

// Helper: wrap coordinate
__device__ float wrap(float x) {
    if (x < 0) return x + WORLD_SIZE;
    if (x >= WORLD_SIZE) return x - WORLD_SIZE;
    return x;
}

// Helper: distance squared
__device__ float dist2(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    // Wrap-around distance
    if (dx > 0.5f) dx = 1.0f - dx;
    if (dy > 0.5f) dy = 1.0f - dy;
    return dx*dx + dy*dy;
}

// Initialize agents and resources
__global__ void init_simulation(int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS + RESOURCES) return;
    
    if (idx < AGENTS) {
        // Initialize agent
        Agent &a = d_agents[idx];
        a.x = lcgf(a.rng);
        a.y = lcgf(a.rng);
        a.vx = lcgf(a.rng) * 0.02f - 0.01f;
        a.vy = lcgf(a.rng) * 0.02f - 0.01f;
        a.energy = 1.0f;
        a.fitness = 0.0f;
        a.arch = idx % ARCHETYPES;
        a.rng = idx * 17 + 12345;
        a.pheromone_memory = 0;
        a.target_x = a.x;
        a.target_y = a.y;
        
        // Set roles based on specialization
        if (specialized) {
            for (int i = 0; i < ARCHETYPES; i++) {
                a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
            }
        } else {
            for (int i = 0; i < ARCHETYPES; i++) {
                a.role[i] = 0.25f;
            }
        }
    } else {
        // Initialize resource
        int res_idx = idx - AGENTS;
        Resource &r = d_resources[res_idx];
        r.x = lcgf(d_agents[0].rng); // Use agent 0's RNG for consistency
        r.y = lcgf(d_agents[0].rng);
        r.value = 0.8f + lcgf(d_agents[0].rng) * 0.4f;
        r.collected = false;
        r.last_collected_tick = -1000;
    }
    
    // Initialize pheromone grid
    if (idx < PHEROMONE_GRID * PHEROMONE_GRID) {
        int i = idx / PHEROMONE_GRID;
        int j = idx % PHEROMONE_GRID;
        for (int arch = 0; arch < ARCHETYPES; arch++) {
            d_pheromone[i][j].strength[arch] = 0.0f;
        }
        d_pheromone[i][j].last_updated = -1000;
    }
}

// Update pheromone grid (decay and diffuse)
__global__ void update_pheromone() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= PHEROMONE_GRID || j >= PHEROMONE_GRID) return;
    
    PheromoneGrid &pg = d_pheromone[i][j];
    int current_tick = d_tick_counter;
    
    // Decay: 5% per tick
    for (int arch = 0; arch < ARCHETYPES; arch++) {
        pg.strength[arch] *= 0.95f;
    }
    
    // Simple diffusion to neighbors (4-way)
    float diffusion_rate = 0.1f;
    for (int arch = 0; arch < ARCHETYPES; arch++) {
        float total = pg.strength[arch];
        int count = 1;
        
        // Check neighbors (wrapped grid)
        int ni = (i + 1) % PHEROMONE_GRID;
        int nj = (j + 1) % PHEROMONE_GRID;
        int pi = (i - 1 + PHEROMONE_GRID) % PHEROMONE_GRID;
        int pj = (j - 1 + PHEROMONE_GRID) % PHEROMONE_GRID;
        
        total += d_pheromone[ni][j].strength[arch];
        total += d_pheromone[pi][j].strength[arch];
        total += d_pheromone[i][nj].strength[arch];
        total += d_pheromone[i][pj].strength[arch];
        count = 5;
        
        pg.strength[arch] = total / count * (1.0f - diffusion_rate) + 
                           pg.strength[arch] * diffusion_rate;
    }
}

// Main simulation tick
__global__ void tick() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent &a = d_agents[idx];
    int current_tick = d_tick_counter;
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with nearby agents
    int similar_count = 0;
    int total_count = 0;
    float similarity_threshold = 0.9f;
    
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent &other = d_agents[i];
        if (dist2(a.x, a.y, other.x, other.y) < 0.04f) {
            total_count++;
            float similarity = 0.0f;
            for (int r = 0; r < ARCHETYPES; r++) {
                similarity += fminf(a.role[r], other.role[r]);
            }
            if (similarity > similarity_threshold) {
                similar_count++;
            }
        }
    }
    
    // Apply anti-convergence drift if too similar
    if (total_count > 0 && (float)similar_count / total_count > 0.5f) {
        int non_dominant = 0;
        float max_role = a.role[0];
        for (int r = 1; r < ARCHETYPES; r++) {
            if (a.role[r] > max_role) {
                max_role = a.role[r];
                non_dominant = r;
            }
        }
        // Randomly choose a different non-dominant role
        non_dominant = (non_dominant + 1 + int(lcgf(a.rng) * (ARCHETYPES-1))) % ARCHETYPES;
        a.role[non_dominant] += lcgf(a.rng) * 0.02f - 0.01f;
        // Renormalize
        float sum = 0.0f;
        for (int r = 0; r < ARCHETYPES; r++) sum += a.role[r];
        for (int r = 0; r < ARCHETYPES; r++) a.role[r] /= sum;
    }
    
    // Pheromone sensing and following (NOVEL MECHANISM)
    int grid_x = int(a.x * PHEROMONE_GRID) % PHEROMONE_GRID;
    int grid_y = int(a.y * PHEROMONE_GRID) % PHEROMONE_GRID;
    
    // Check if we should follow pheromone or explore randomly
    float pheromone_strength = d_pheromone[grid_x][grid_y].strength[a.arch];
    
    if (a.pheromone_memory > 0) {
        a.pheromone_memory--;
        // Move toward remembered target
        float dx = a.target_x - a.x;
        float dy = a.target_y - a.y;
        if (dx > 0.5f) dx = dx - 1.0f;
        if (dx < -0.5f) dx = dx + 1.0f;
        if (dy > 0.5f) dy = dy - 1.0f;
        if (dy < -0.5f) dy = dy + 1.0f;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist > 0.001f) {
            a.vx = dx / dist * 0.01f;
            a.vy = dy / dist * 0.01f;
        }
    } else if (pheromone_strength > 0.1f && lcgf(a.rng) < a.role[a.arch] * 0.5f) {
        // Strong pheromone detected, follow gradient
        float best_strength = pheromone_strength;
        int best_x = grid_x;
        int best_y = grid_y;
        
        // Check 3x3 neighborhood for strongest pheromone
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = (grid_x + dx + PHEROMONE_GRID) % PHEROMONE_GRID;
                int ny = (grid_y + dy + PHEROMONE_GRID) % PHEROMONE_GRID;
                float strength = d_pheromone[nx][ny].strength[a.arch];
                if (strength > best_strength) {
                    best_strength = strength;
                    best_x = nx;
                    best_y = ny;
                }
            }
        }
        
        if (best_x != grid_x || best_y != grid_y) {
            // Set target to center of that grid cell
            a.target_x = (best_x + 0.5f) / PHEROMONE_GRID;
            a.target_y = (best_y + 0.5f) / PHEROMONE_GRID;
            a.pheromone_memory = 10; // Follow for 10 ticks
        }
    } else {
        // Normal movement based on role
        a.vx += (lcgf(a.rng) - 0.5f) * 0.002f;
        a.vy += (lcgf(a.rng) - 0.5f) * 0.002f;
    }
    
    // Velocity limits and position update
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.02f) {
        a.vx = a.vx / speed * 0.02f;
        a.vy = a.vy / speed * 0.02f;
    }
    
    a.x = wrap(a.x + a.vx);
    a.y = wrap(a.y + a.vy);
    
    // Role-specific behaviors
    float detect_range = 0.03f + a.role[a.arch] * 0.04f;
    float grab_range = 0.02f + a.role[a.arch] * 0.02f;
    float comm_range = 0.06f;
    
    // Resource collection
    int nearest_res = -1;
    float nearest_dist2 = 1e6f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource &r = d_resources[i];
        if (r.collected && current_tick - r.last_collected_tick < 50) {
            continue; // Recently collected
        }
        
        float d2 = dist2(a.x, a.y, r.x, r.y);
        if (d2 < detect_range * detect_range) {
            if (d2 < nearest_dist2) {
                nearest_dist2 = d2;
                nearest_res = i;
            }
        }
    }
    
    // Collect resource if in grab range
    if (nearest_res >= 0 && nearest_dist2 < grab_range * grab_range) {
        Resource &r = d_resources[nearest_res];
        if (!r.collected || current_tick - r.last_collected_tick >= 50) {
            // Collection bonus based on role specialization
            float bonus = 1.0f + a.role[a.arch] * 0.5f;
            a.energy += r.value * bonus;
            a.fitness += r.value * bonus;
            r.collected = true;
            r.last_collected_tick = current_tick;
            
            // NOVEL: Deposit pheromone at resource location
            int px = int(r.x * PHEROMONE_GRID) % PHEROMONE_GRID;
            int py = int(r.y * PHEROMONE_GRID) % PHEROMONE_GRID;
            atomicAdd(&d_pheromone[px][py].strength[a.arch], 1.0f);
            d_pheromone[px][py].last_updated = current_tick;
        }
    }
    
    // Communication (broadcast nearest resource)
    if (nearest_res >= 0 && lcgf(a.rng) < 0.1f) {
        Resource &r = d_resources[nearest_res];
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = d_agents[i];
            if (dist2(a.x, a.y, other.x, other.y) < comm_range * comm_range) {
                // Same archetype coupling is stronger
                float coupling = (a.arch == other.arch) ? 0.02f : 0.002f;
                // Influence other's velocity toward resource
                float dx = r.x - other.x;
                float dy = r.y - other.y;
                if (dx > 0.5f) dx = dx - 1.0f;
                if (dx < -0.5f) dx = dx + 1.0f;
                if (dy > 0.5f) dy = dy - 1.0f;
                if (dy < -0.5f) dy = dy + 1.0f;
                
                float dist = sqrtf(dx*dx + dy*dy);
                if (dist > 0.001f) {
                    other.vx += dx / dist * coupling;
                    other.vy += dy / dist * coupling;
                }
            }
        }
    }
    
    // Territory defense bonus
    int defenders_nearby = 0;
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent &other = d_agents[i];
        if (a.arch == other.arch && dist2(a.x, a.y, other.x, other.y) < 0.04f) {
            defenders_nearby++;
        }
    }
    
    // Defense bonus: 20% per nearby defender
    if (defenders_nearby > 0) {
        a.energy *= 1.0f + defenders_nearby * 0.2f;
    }
    
    // Random perturbation (defenders resist)
    if (lcgf(a.rng) < 0.001f) {
        float resistance = a.role[a.arch]; // Specialists resist better
        if (lcgf(a.rng) > resistance * 0.5f) {
            a.vx +=