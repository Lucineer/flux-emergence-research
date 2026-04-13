/*
CUDA Simulation Experiment v66: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination beyond basic communication,
            increasing the specialist advantage ratio to >1.8x (vs v8's 1.61x).
Mechanism: When an agent collects a resource, it deposits pheromone (strength=1.0).
           Pheromones decay by 0.01/tick. Agents can detect pheromones within 0.08 range.
           Specialists get +0.3 bonus to pheromone detection in their primary role.
Baseline: Includes all v8 confirmed mechanisms (scarcity, territory, comms).
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
const int GRID_SIZE = 256;
const float CELL_SIZE = 1.0f / GRID_SIZE;
const float PHEROMONE_DECAY = 0.01f;
const float PHEROMONE_DEPOSIT = 1.0f;
const float PHEROMONE_DETECT_RANGE = 0.08f;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent structure
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];
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

// Pheromone grid structure
struct PheromoneGrid {
    float trail[GRID_SIZE][GRID_SIZE];
};

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize pheromone grid
__global__ void initPheromoneKernel(PheromoneGrid *grid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= GRID_SIZE * GRID_SIZE) return;
    
    int i = idx / GRID_SIZE;
    int j = idx % GRID_SIZE;
    grid->trail[i][j] = 0.0f;
}

// Decay pheromones
__global__ void decayPheromoneKernel(PheromoneGrid *grid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= GRID_SIZE * GRID_SIZE) return;
    
    int i = idx / GRID_SIZE;
    int j = idx % GRID_SIZE;
    grid->trail[i][j] = fmaxf(0.0f, grid->trail[i][j] - PHEROMONE_DECAY);
}

// Deposit pheromone at location
__device__ void depositPheromone(PheromoneGrid *grid, float x, float y) {
    int gx = min(GRID_SIZE - 1, max(0, (int)(x / CELL_SIZE)));
    int gy = min(GRID_SIZE - 1, max(0, (int)(y / CELL_SIZE)));
    atomicAdd(&grid->trail[gx][gy], PHEROMONE_DEPOSIT);
}

// Read pheromone at location
__device__ float readPheromone(PheromoneGrid *grid, float x, float y) {
    int gx = min(GRID_SIZE - 1, max(0, (int)(x / CELL_SIZE)));
    int gy = min(GRID_SIZE - 1, max(0, (int)(y / CELL_SIZE)));
    return grid->trail[gx][gy];
}

// Initialize agents
__global__ void initAgents(Agent *agents, int specialist) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    a.rng = idx * 17 + 12345;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = lcgf(a.rng) * 0.02f - 0.01f;
    a.vy = lcgf(a.rng) * 0.02f - 0.01f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    
    if (specialist) {
        // Specialized agents: primary role = 0.7, others = 0.1
        a.arch = idx % 4;
        for (int i = 0; i < 4; i++) {
            a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles = 0.25
        a.arch = ARCH_EXPLORER;
        for (int i = 0; i < 4; i++) {
            a.role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void initResources(Resource *resources) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RES_COUNT) return;
    
    Resource &r = resources[idx];
    unsigned int rng = idx * 19 + 54321;
    r.x = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
    r.y = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
    r.value = 0.5f + (lcg(rng) & 0xFFFFFF) / 16777216.0f * 0.5f;
    r.collected = 0;
}

// Main simulation tick
__global__ void tickKernel(Agent *agents, Resource *resources, PheromoneGrid *grid, 
                          int tick, int *perturb_flag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Apply perturbation every 50 ticks (defenders resist)
    if (tick % 50 == 0 && tick > 0) {
        if (a.arch != ARCH_DEFENDER || lcgf(a.rng) > 0.7f) {
            a.energy *= 0.5f;
            *perturb_flag = 1;
        }
    }
    
    // Anti-convergence: check similarity with random agent
    int other_idx = lcg(a.rng) % AGENT_COUNT;
    Agent &other = agents[other_idx];
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a.role[i] - other.role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Apply drift to non-dominant role
        int drift_role = (a.arch + 1 + (lcg(a.rng) % 3)) % 4;
        a.role[drift_role] += (lcgf(a.rng) * 0.02f - 0.01f);
        a.role[drift_role] = fmaxf(0.05f, fminf(0.9f, a.role[drift_role]));
    }
    
    // Movement with pheromone influence
    float fx = 0.0f, fy = 0.0f;
    
    // Read pheromone gradient (specialists get bonus in their primary role)
    float pheromone_strength = 0.0f;
    float detect_bonus = (a.arch == ARCH_EXPLORER) ? 0.3f : 0.0f;
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            float sample_x = a.x + dx * PHEROMONE_DETECT_RANGE * 0.5f;
            float sample_y = a.y + dy * PHEROMONE_DETECT_RANGE * 0.5f;
            float p = readPheromone(grid, sample_x, sample_y);
            pheromone_strength += p;
            
            if (p > 0.1f) {
                float weight = p * (a.role[ARCH_EXPLORER] + detect_bonus);
                fx += dx * weight;
                fy += dy * weight;
            }
        }
    }
    
    // Normal movement based on role distribution
    a.vx += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[ARCH_EXPLORER];
    a.vy += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[ARCH_EXPLORER];
    
    // Add pheromone influence
    float pmag = sqrtf(fx*fx + fy*fy);
    if (pmag > 0.001f) {
        a.vx += fx / pmag * 0.005f * a.role[ARCH_EXPLORER];
        a.vy += fy / pmag * 0.005f * a.role[ARCH_EXPLORER];
    }
    
    // Velocity limits
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.02f) {
        a.vx *= 0.02f / speed;
        a.vy *= 0.02f / speed;
    }
    
    // Update position (toroidal world)
    a.x += a.vx;
    a.y += a.vy;
    if (a.x < 0.0f) a.x += 1.0f;
    if (a.x >= 1.0f) a.x -= 1.0f;
    if (a.y < 0.0f) a.y += 1.0f;
    if (a.y >= 1.0f) a.y -= 1.0f;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    // Detection range based on explorer role
    float detect_range = 0.03f + a.role[ARCH_EXPLORER] * 0.04f;
    
    for (int i = 0; i < RES_COUNT; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        // Toroidal distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    if (best_res != -1) {
        Resource &r = resources[best_res];
        float grab_range = 0.02f + a.role[ARCH_COLLECTOR] * 0.02f;
        
        if (best_dist < grab_range) {
            // Collect resource
            float value = r.value;
            // Collector bonus
            value *= 1.0f + a.role[ARCH_COLLECTOR] * 0.5f;
            
            // Territory bonus from nearby defenders of same archetype
            int defender_count = 0;
            for (int j = 0; j < AGENT_COUNT; j++) {
                if (j == idx) continue;
                Agent &other = agents[j];
                if (other.arch != a.arch) continue;
                
                float dx2 = other.x - a.x;
                float dy2 = other.y - a.y;
                if (dx2 > 0.5f) dx2 -= 1.0f;
                if (dx2 < -0.5f) dx2 += 1.0f;
                if (dy2 > 0.5f) dy2 -= 1.0f;
                if (dy2 < -0.5f) dy2 += 1.0f;
                
                float dist2 = sqrtf(dx2*dx2 + dy2*dy2);
                if (dist2 < 0.1f && other.arch == ARCH_DEFENDER) {
                    defender_count++;
                }
            }
            
            value *= 1.0f + defender_count * 0.2f;
            
            a.energy += value;
            a.fitness += value;
            r.collected = 1;
            
            // DEPOSIT PHEROMONE (novel mechanism)
            depositPheromone(grid, r.x, r.y);
        }
    }
    
    // Communication (broadcast nearest resource location)
    if (a.role[ARCH_COMMUNICATOR] > 0.3f) {
        float comm_range = 0.06f;
        for (int j = 0; j < AGENT_COUNT; j++) {
            if (j == idx) continue;
            Agent &other = agents[j];
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx > 0.5f) dx -= 1.0f;
            if (dx < -0.5f) dx += 1.0f;
            if (dy > 0.5f) dy -= 1.0f;
            if (dy < -0.5f) dy += 1.0f;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < comm_range) {
                // Simple influence: adjust velocity toward known resources
                if (best_res != -1) {
                    Resource &r = resources[best_res];
                    float rdx = r.x - other.x;
                    float rdy = r.y - other.y;
                    if (rdx > 0.5f) rdx -= 1.0f;
                    if (rdx < -0.5f) rdx += 1.0f;
                    if (rdy > 0.5f) rdy -= 1.0f;
                    if (rdy < -0.5f) rdy += 1.0f;
                    
                    float rdist = sqrtf(rdx*rdx + rdy*rdy);
                    if (rdist > 0.001f) {
                        other.vx += rdx / rdist * 0.003f * a.role[ARCH_COMMUNICATOR];
                        other.vy += rdy / rdist * 0.003f * a.role[ARCH_COMMUNICATOR];
                    }
                }
            }
        }
    }
    
    // Energy limits
    if (a.energy > 2.0f) a.energy = 2.0f;
    if (a.energy < 0.0f) a.energy = 0.0f;
}

int main() {
    printf("Experiment v66: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone markers enhance specialist coordination\n");
    printf("Prediction: Specialist advantage >1.8x (vs v8's 1.61x)\n\n");
    
    // Allocate memory
    Agent *d_agents_spec, *d_agents_uniform;
    Resource *d_resources_spec, *d_resources_uniform;
    PheromoneGrid *d_grid_spec, *d_grid_uniform;
    int *d_perturb;
    
    cudaMalloc(&d_agents_spec, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_resources_spec, RES_COUNT * sizeof(Resource));
    cudaMalloc(&d_resources_uniform, RES_COUNT * sizeof(Resource));
    cudaMalloc(&d_grid_spec, sizeof(PheromoneGrid));
    cudaMalloc(&d_grid_uniform, sizeof(PheromoneGrid));
    cudaMalloc(&d_perturb, sizeof(int));
    
    Agent *h_agents_spec = new Agent[AGENT_COUNT];
    Agent *h_agents_uniform = new Agent[AGENT_COUNT];
    
    // Initialize
    int blocks = (AGENT_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int res_blocks = (RES_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_blocks = (GRID_SIZE*GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    initPheromoneKernel<<<grid_blocks, BLOCK_SIZE>>>(d_grid_spec);
    initPheromoneKernel<<<grid_blocks, BLOCK_SIZE>>>(d_grid_un