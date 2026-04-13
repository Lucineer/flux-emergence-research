
/*
CUDA Simulation Experiment v80: STIGMERGY TRAILS
Testing: Pheromone trails at resource locations that decay over time
Prediction: Stigmergy will amplify specialist advantage (ratio > 1.61x) by creating 
           persistent information about resource locations, reducing search costs.
Baseline: v8 confirmed mechanisms (scarcity, territory, comms) + anti-convergence
Novel: Agents deposit pheromone when collecting resources, others follow gradients
Control: Uniform agents (all roles=0.25) vs Specialized (role[arch]=0.7)
Expected: Specialists better at using trails due to role coordination
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const float WORLD_SIZE = 1.0f;
const float MIN_DIST = 0.0001f;

// Pheromone grid
const int GRID_SIZE = 256;
const float CELL_SIZE = WORLD_SIZE / GRID_SIZE;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_DEPOSIT = 1.0f;
const float TRAIL_FOLLOW_STRENGTH = 0.3f;

// Agent roles
enum RoleIndex { EXPLORE = 0, COLLECT = 1, COMMUNICATE = 2, DEFEND = 3 };

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Resource
struct Resource {
    float x, y;
    float value;
    bool collected;
    int last_visit;
};

// Pheromone cell
struct Pheromone {
    float intensity[ARCHETYPES];
};

// Agent
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPES];
    float fitness;
    int arch;
    unsigned int rng;
    
    // Memory for communication
    float comm_target_x;
    float comm_target_y;
    bool has_target;
};

// Initialize agents
__global__ void init_agents(Agent* agents, int num_agents, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    unsigned int local_seed = seed + idx * 137;
    agents[idx].x = lcgf(&local_seed) * WORLD_SIZE;
    agents[idx].y = lcgf(&local_seed) * WORLD_SIZE;
    agents[idx].vx = (lcgf(&local_seed) - 0.5f) * 0.02f;
    agents[idx].vy = (lcgf(&local_seed) - 0.5f) * 0.02f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].rng = local_seed;
    agents[idx].arch = idx % ARCHETYPES;
    agents[idx].has_target = false;
    
    // Specialized agents: strong in one role, weak in others
    if (idx < num_agents / 2) {
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = (i == agents[idx].arch) ? 0.7f : 0.1f;
        }
    } 
    // Uniform control agents
    else {
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
    
    // Normalize
    float sum = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) sum += agents[idx].role[i];
    for (int i = 0; i < ARCHETYPES; i++) agents[idx].role[i] /= sum;
}

// Initialize resources
__global__ void init_resources(Resource* resources, int num_res, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_res) return;
    
    unsigned int local_seed = seed + idx * 7919;
    resources[idx].x = lcgf(&local_seed) * WORLD_SIZE;
    resources[idx].y = lcgf(&local_seed) * WORLD_SIZE;
    resources[idx].value = 0.5f + lcgf(&local_seed) * 0.5f;
    resources[idx].collected = false;
    resources[idx].last_visit = -1000;
}

// Initialize pheromone grid
__global__ void init_pheromone(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    for (int a = 0; a < ARCHETYPES; a++) {
        grid[idx].intensity[a] = 0.0f;
    }
}

// Decay pheromone
__global__ void decay_pheromone(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    for (int a = 0; a < ARCHETYPES; a++) {
        grid[idx].intensity[a] *= PHEROMONE_DECAY;
    }
}

// Get grid cell from position
__device__ int get_grid_cell(float x, float y) {
    int gx = (int)(x / CELL_SIZE);
    int gy = (int)(y / CELL_SIZE);
    gx = max(0, min(GRID_SIZE - 1, gx));
    gy = max(0, min(GRID_SIZE - 1, gy));
    return gy * GRID_SIZE + gx;
}

// Deposit pheromone
__device__ void deposit_pheromone(Pheromone* grid, float x, float y, int arch) {
    int cell = get_grid_cell(x, y);
    atomicAdd(&grid[cell].intensity[arch], PHEROMONE_DEPOSIT);
}

// Sample pheromone gradient
__device__ void get_pheromone_gradient(Pheromone* grid, float x, float y, int arch, 
                                      float* grad_x, float* grad_y) {
    int cell = get_grid_cell(x, y);
    float center = grid[cell].intensity[arch];
    
    // Sample neighbors
    float left = (x > CELL_SIZE) ? grid[get_grid_cell(x - CELL_SIZE, y)].intensity[arch] : center;
    float right = (x < WORLD_SIZE - CELL_SIZE) ? grid[get_grid_cell(x + CELL_SIZE, y)].intensity[arch] : center;
    float up = (y > CELL_SIZE) ? grid[get_grid_cell(x, y - CELL_SIZE)].intensity[arch] : center;
    float down = (y < WORLD_SIZE - CELL_SIZE) ? grid[get_grid_cell(x, y + CELL_SIZE)].intensity[arch] : center;
    
    *grad_x = (right - left) / (2.0f * CELL_SIZE);
    *grad_y = (down - up) / (2.0f * CELL_SIZE);
    
    // Normalize
    float len = sqrtf(*grad_x * *grad_x + *grad_y * *grad_y);
    if (len > 0.001f) {
        *grad_x /= len;
        *grad_y /= len;
    }
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromone, 
                     int num_agents, int num_res, int tick_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with nearby agents
    int similar_count = 0;
    int total_count = 0;
    for (int i = 0; i < min(10, num_agents); i++) {
        int j = (idx + i) % num_agents;
        if (j == idx) continue;
        
        Agent& other = agents[j];
        float dx = a.x - other.x;
        float dy = a.y - other.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.1f) {
            total_count++;
            float similarity = 0.0f;
            for (int r = 0; r < ARCHETYPES; r++) {
                similarity += fminf(a.role[r], other.role[r]);
            }
            if (similarity > 0.9f) similar_count++;
        }
    }
    
    // Apply anti-convergence drift
    if (total_count > 0 && (float)similar_count / total_count > 0.5f) {
        int drift_role = (a.arch + tick_id) % ARCHETYPES;
        a.role[drift_role] += (lcgf(&a.rng) - 0.5f) * 0.01f;
        a.role[drift_role] = max(0.01f, min(0.99f, a.role[drift_role]));
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < ARCHETYPES; i++) sum += a.role[i];
        for (int i = 0; i < ARCHETYPES; i++) a.role[i] /= sum;
    }
    
    // Role-based behavior
    float explore_strength = a.role[EXPLORE];
    float collect_strength = a.role[COLLECT];
    float comm_strength = a.role[COMMUNICATE];
    float defend_strength = a.role[DEFEND];
    
    // 1. Explore: random walk with pheromone following
    float explore_dx = (lcgf(&a.rng) - 0.5f) * 0.04f;
    float explore_dy = (lcgf(&a.rng) - 0.5f) * 0.04f;
    
    // Follow pheromone gradient of own archetype
    float trail_dx = 0.0f, trail_dy = 0.0f;
    get_pheromone_gradient(pheromone, a.x, a.y, a.arch, &trail_dx, &trail_dy);
    
    explore_dx += trail_dx * TRAIL_FOLLOW_STRENGTH * explore_strength;
    explore_dy += trail_dy * TRAIL_FOLLOW_STRENGTH * explore_strength;
    
    // 2. Collect: move toward known resources
    float collect_dx = 0.0f, collect_dy = 0.0f;
    if (a.has_target) {
        float dx = a.comm_target_x - a.x;
        float dy = a.comm_target_y - a.y;
        float dist = sqrtf(dx*dx + dy*dy) + MIN_DIST;
        collect_dx = dx / dist * 0.03f;
        collect_dy = dy / dist * 0.03f;
    }
    
    // 3. Communication: share locations
    if (comm_strength > 0.3f && tick_id % 10 == 0) {
        // Find nearest resource
        float best_dist = 1e6f;
        float best_x = 0.0f, best_y = 0.0f;
        for (int i = 0; i < min(20, num_res); i++) {
            int ridx = (idx * 17 + i) % num_res;
            Resource& r = resources[ridx];
            if (r.collected) continue;
            
            float dx = a.x - r.x;
            float dy = a.y - r.y;
            float dist = dx*dx + dy*dy;
            
            if (dist < best_dist && dist < 0.04f) {
                best_dist = dist;
                best_x = r.x;
                best_y = r.y;
            }
        }
        
        if (best_dist < 1e5f) {
            a.has_target = true;
            a.comm_target_x = best_x;
            a.comm_target_y = best_y;
            
            // Broadcast to nearby agents
            for (int i = 0; i < min(5, num_agents); i++) {
                int j = (idx + i * 31) % num_agents;
                if (j == idx) continue;
                
                Agent& other = agents[j];
                float dx = a.x - other.x;
                float dy = a.y - other.y;
                if (dx*dx + dy*dy < 0.0036f) {  // 0.06^2
                    other.has_target = true;
                    other.comm_target_x = best_x;
                    other.comm_target_y = best_y;
                }
            }
        }
    }
    
    // 4. Defend: territory behavior
    float defend_dx = 0.0f, defend_dy = 0.0f;
    int defender_count = 0;
    for (int i = 0; i < min(10, num_agents); i++) {
        int j = (idx + i * 7) % num_agents;
        if (j == idx) continue;
        
        Agent& other = agents[j];
        if (other.arch != a.arch) continue;
        
        float dx = a.x - other.x;
        float dy = a.y - other.y;
        float dist2 = dx*dx + dy*dy;
        
        if (dist2 < 0.01f) {  // 0.1^2
            defender_count++;
            // Mild cohesion
            defend_dx -= dx * 0.01f;
            defend_dy -= dy * 0.01f;
        }
    }
    
    // Defender bonus: more defenders = more collection efficiency
    float defender_bonus = 1.0f + defender_count * 0.2f;
    
    // Combine behaviors
    a.vx = explore_dx * explore_strength + 
           collect_dx * collect_strength + 
           defend_dx * defend_strength;
    a.vy = explore_dy * explore_strength + 
           collect_dy * collect_strength + 
           defend_dy * defend_strength;
    
    // Normalize velocity
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.03f) {
        a.vx = a.vx / speed * 0.03f;
        a.vy = a.vy / speed * 0.03f;
    }
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World boundaries
    if (a.x < 0) { a.x = 0; a.vx = fabsf(a.vx); }
    if (a.x > WORLD_SIZE) { a.x = WORLD_SIZE; a.vx = -fabsf(a.vx); }
    if (a.y < 0) { a.y = 0; a.vy = fabsf(a.vy); }
    if (a.y > WORLD_SIZE) { a.y = WORLD_SIZE; a.vy = -fabsf(a.vy); }
    
    // Resource collection
    for (int i = 0; i < min(10, num_res); i++) {
        int ridx = (idx * 13 + i + tick_id) % num_res;
        Resource& r = resources[ridx];
        
        if (r.collected) continue;
        
        float dx = a.x - r.x;
        float dy = a.y - r.y;
        float dist2 = dx*dx + dy*dy;
        
        float grab_range = 0.02f + collect_strength * 0.02f;
        
        if (dist2 < grab_range * grab_range) {
            // Collect resource
            float value = r.value * defender_bonus;
            if (collect_strength > 0.5f) value *= 1.5f;  // Specialist bonus
            
            a.energy += value;
            a.fitness += value;
            r.collected = true;
            r.last_visit = tick_id;
            
            // Deposit pheromone at resource location
            deposit_pheromone(pheromone, r.x, r.y, a.arch);
            
            // Clear communication target
            a.has_target = false;
            break;
        }
    }
    
    // Resource respawn (scarcity)
    if (tick_id % 50 == 0) {
        for (int i = 0; i < min(5, num_res); i++) {
            int ridx = (idx * 19 + i) % num_res;
            Resource& r = resources[ridx];
            
            if (r.collected && (tick_id - r.last_visit) > 10) {
                r.collected = false;
                r.x = lcgf(&a.rng) * WORLD_SIZE;
                r.y = lcgf(&a.rng) * WORLD_SIZE;
                r.value = 0.5f + lcgf(&a.rng) * 0.5f;
            }
        }
    }
    
    // Perturbation (every 100 ticks)
    if (tick_id % 100 == 50) {
        float resist =