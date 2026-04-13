// CUDA Simulation Experiment v94: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist advantage
// Prediction: Pheromones will amplify specialist advantage (ratio > 1.61x) by creating
//             persistent information that complements communication
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Novelty: Agents leave pheromone markers at collected resources that decay over time
//          Other agents can detect pheromone intensity within their detection range

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const int PHEROMONE_GRID = 256; // 256x256 grid for pheromone field
const float WORLD_SIZE = 1.0f;

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Agent struct
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role strengths: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Archetype 0-3
    unsigned int rng;     // RNG state
};

// Resource struct
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone struct for grid-based field
struct PheromoneGrid {
    float intensity[PHEROMONE_GRID * PHEROMONE_GRID];
    float arch_type[PHEROMONE_GRID * PHEROMONE_GRID]; // Which archetype left it
};

// Initialize agents kernel
__global__ void init_agents(Agent *agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    a.rng = idx * 17 + 12345;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = (lcgf(a.rng) - 0.5f) * 0.02f;
    a.vy = (lcgf(a.rng) - 0.5f) * 0.02f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % ARCHETYPES;
    
    if (specialized) {
        // Specialized: strong in own archetype's role (0.7), weak in others (0.1)
        for (int i = 0; i < 4; i++) {
            a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles equal
        for (int i = 0; i < 4; i++) {
            a.role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = resources[idx];
    unsigned int rng = idx * 29 + 54321;
    r.x = lcgf(rng);
    r.y = lcgf(rng);
    r.value = 0.5f + lcgf(rng) * 0.5f; // 0.5-1.0
    r.collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(PheromoneGrid *grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    grid->intensity[idx] = 0.0f;
    grid->arch_type[idx] = -1.0f;
}

// Update pheromone decay and diffusion
__global__ void update_pheromones(PheromoneGrid *grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    // Decay
    grid->intensity[idx] *= 0.95f; // 5% decay per tick
    
    // Simple diffusion to neighbors (3x3 kernel)
    float diff = 0.0f;
    int x = idx % PHEROMONE_GRID;
    int y = idx / PHEROMONE_GRID;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < PHEROMONE_GRID && ny >= 0 && ny < PHEROMONE_GRID) {
                int nidx = ny * PHEROMONE_GRID + nx;
                diff += grid->intensity[nidx];
            }
        }
    }
    
    // Blend with diffusion
    grid->intensity[idx] = 0.8f * grid->intensity[idx] + 0.2f * (diff / 9.0f);
    
    // Clear very faint pheromones
    if (grid->intensity[idx] < 0.01f) {
        grid->intensity[idx] = 0.0f;
        grid->arch_type[idx] = -1.0f;
    }
}

// Main simulation kernel
__global__ void tick_kernel(Agent *agents, Resource *resources, PheromoneGrid *pheromones, 
                           int tick, int *resource_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with archetype pattern
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        float expected = (i == a.arch) ? 0.7f : 0.1f;
        similarity += 1.0f - fabsf(a.role[i] - expected);
    }
    similarity /= 4.0f;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant roles
        int drift_target = (a.arch + 1 + (int)(lcgf(a.rng) * 3)) % 4;
        a.role[drift_target] += (lcgf(a.rng) - 0.5f) * 0.01f;
        
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int i = 0; i < 4; i++) a.role[i] /= sum;
    }
    
    // Movement based on explore role
    float explore_strength = a.role[0];
    a.vx += (lcgf(a.rng) - 0.5f) * 0.02f * explore_strength;
    a.vy += (lcgf(a.rng) - 0.5f) * 0.02f * explore_strength;
    
    // Velocity damping
    a.vx *= 0.98f;
    a.vy *= 0.98f;
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World boundaries (bouncing)
    if (a.x < 0) { a.x = 0; a.vx = fabsf(a.vx); }
    if (a.x > WORLD_SIZE) { a.x = WORLD_SIZE; a.vx = -fabsf(a.vx); }
    if (a.y < 0) { a.y = 0; a.vy = fabsf(a.vy); }
    if (a.y > WORLD_SIZE) { a.y = WORLD_SIZE; a.vy = -fabsf(a.vy); }
    
    // Pheromone detection
    int grid_x = (int)(a.x * PHEROMONE_GRID);
    int grid_y = (int)(a.y * PHEROMONE_GRID);
    grid_x = max(0, min(PHEROMONE_GRID - 1, grid_x));
    grid_y = max(0, min(PHEROMONE_GRID - 1, grid_y));
    
    float pheromone_intensity = 0.0f;
    float pheromone_arch = -1.0f;
    
    // Check 3x3 area around agent
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = grid_x + dx;
            int ny = grid_y + dy;
            if (nx >= 0 && nx < PHEROMONE_GRID && ny >= 0 && ny < PHEROMONE_GRID) {
                int pidx = ny * PHEROMONE_GRID + nx;
                if (pheromones->intensity[pidx] > pheromone_intensity) {
                    pheromone_intensity = pheromones->intensity[pidx];
                    pheromone_arch = pheromones->arch_type[pidx];
                }
            }
        }
    }
    
    // If strong pheromone detected and matches archetype, move toward it
    if (pheromone_intensity > 0.3f && fabsf(pheromone_arch - a.arch) < 0.5f) {
        // Move toward pheromone gradient
        float target_x = (grid_x + 0.5f) / PHEROMONE_GRID;
        float target_y = (grid_y + 0.5f) / PHEROMONE_GRID;
        a.vx += (target_x - a.x) * 0.01f * explore_strength;
        a.vy += (target_y - a.y) * 0.01f * explore_strength;
    }
    
    // Resource interaction
    float collect_range = 0.02f + a.role[1] * 0.02f; // 0.02-0.04
    float detect_range = 0.03f + a.role[0] * 0.04f;  // 0.03-0.07
    
    float best_dist = 1e6;
    int best_res = -1;
    
    // Find nearest resource
    for (int i = 0; i < RESOURCES; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    // If resource in collect range, collect it
    if (best_res != -1 && best_dist < collect_range) {
        Resource &r = resources[best_res];
        
        // Collection bonus based on collect role
        float bonus = 1.0f + a.role[1] * 0.5f; // Up to 50% bonus
        
        // Territory bonus: check for nearby defenders of same archetype
        float territory_bonus = 1.0f;
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent &other = agents[j];
            if (other.arch != a.arch) continue;
            
            float odx = other.x - a.x;
            float ody = other.y - a.y;
            float odist = sqrtf(odx*odx + ody*ody);
            
            if (odist < 0.1f && other.role[3] > 0.3f) { // Defender nearby
                territory_bonus += 0.2f; // 20% per defender
            }
        }
        
        float energy_gain = r.value * bonus * territory_bonus;
        a.energy += energy_gain;
        a.fitness += energy_gain;
        
        // Leave pheromone at collected resource location
        int pgrid_x = (int)(r.x * PHEROMONE_GRID);
        int pgrid_y = (int)(r.y * PHEROMONE_GRID);
        pgrid_x = max(0, min(PHEROMONE_GRID - 1, pgrid_x));
        pgrid_y = max(0, min(PHEROMONE_GRID - 1, pgrid_y));
        
        int pidx = pgrid_y * PHEROMONE_GRID + pgrid_x;
        pheromones->intensity[pidx] = 1.0f; // Strong pheromone
        pheromones->arch_type[pidx] = (float)a.arch;
        
        r.collected = 1;
        atomicAdd(resource_counter, 1);
    }
    
    // Communication role: broadcast resource locations
    if (a.role[2] > 0.3f && best_res != -1) {
        float comm_range = 0.06f;
        Resource &r = resources[best_res];
        
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent &other = agents[j];
            if (other.arch != a.arch) continue;
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < comm_range) {
                // Attract toward resource
                float influence = a.role[2] * 0.02f;
                other.vx += (r.x - other.x) * influence;
                other.vy += (r.y - other.y) * influence;
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick % 50 == 25) {
        // Defenders resist perturbation
        float resistance = a.role[3] * 0.5f; // Up to 50% resistance
        if (lcgf(a.rng) > resistance) {
            a.energy *= 0.5f; // Halve energy
            a.vx += (lcgf(a.rng) - 0.5f) * 0.1f;
            a.vy += (lcgf(a.rng) - 0.5f) * 0.1f;
        }
    }
    
    // Resource respawn (every 50 ticks)
    if (tick % 50 == 0 && tick > 0) {
        for (int i = 0; i < RESOURCES; i++) {
            resources[i].collected = 0;
        }
        *resource_counter = 0;
    }
}

int main() {
    printf("Experiment v94: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone trails + scarcity + territory + comms\n");
    printf("Prediction: Pheromones amplify specialist advantage > 1.61x\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RESOURCES, TICKS);
    
    // Allocate device memory
    Agent *d_agents_spec, *d_agents_uniform;
    Resource *d_resources_spec, *d_resources_uniform;
    PheromoneGrid *d_pheromones_spec, *d_pheromones_uniform;
    int *d_resource_counter_spec, *d_resource_counter_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources_spec, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_resources_uniform, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, sizeof(PheromoneGrid));
    cudaMalloc(&d_pheromones_uniform, sizeof(PheromoneGrid));
    cudaMalloc(&d_resource_counter_spec, sizeof(int));
    cudaMalloc(&d_resource_counter_uniform, sizeof(int));
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_pheromone((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    init_agents<<<grid_spec, block>>>(d_agents_spec, 1); // Specialized
    init_agents<<<grid_spec, block>>>(d_agents_uniform, 0); // Uniform
    
    init_resources<<<grid_res, block>>>(d_resources_spec);
    init_resources