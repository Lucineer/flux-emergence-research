// CUDA Simulation Experiment v46: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone trails at resource locations that decay over time
// Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents
// Expected: Specialists will use trails more efficiently, widening the 1.61x baseline advantage

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

// Agent struct
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles: explore, collect, communicate, defend
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

// Pheromone struct for stigmergy
struct Pheromone {
    float strength[ARCHETYPES]; // Pheromone strength per archetype
    float decay_rate;   // Decay per tick
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
__global__ void init_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < PHEROMONE_GRID * PHEROMONE_GRID) {
        for (int a = 0; a < ARCHETYPES; a++) {
            pheromones[idx].strength[a] = 0.0f;
        }
        pheromones[idx].decay_rate = 0.95f;
    }
}

// Decay pheromones each tick
__global__ void decay_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < PHEROMONE_GRID * PHEROMONE_GRID) {
        for (int a = 0; a < ARCHETYPES; a++) {
            pheromones[idx].strength[a] *= pheromones[idx].decay_rate;
        }
    }
}

// Initialize agents
__global__ void init_agents(Agent *agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    a.rng = idx * 17 + 12345;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = lcgf(a.rng) * 0.02f - 0.01f;
    a.vy = lcgf(a.rng) * 0.02f - 0.01f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % ARCHETYPES;
    
    if (specialized) {
        // Specialized agents: strong in one role (0.7), weak in others (0.1)
        for (int i = 0; i < 4; i++) {
            a.role[i] = 0.1f;
        }
        a.role[a.arch] = 0.7f;
    } else {
        // Uniform control: all roles equal
        for (int i = 0; i < 4; i++) {
            a.role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void init_resources(Resource *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = resources[idx];
    unsigned int rng = idx * 19 + 54321;
    r.x = (lcg(rng) & 0xFF) / 255.0f;
    r.y = (lcg(rng) & 0xFF) / 255.0f;
    r.value = 0.5f + lcgf(rng) * 0.5f;
    r.collected = 0;
}

// Main simulation kernel
__global__ void tick(Agent *agents, Resource *resources, Pheromone *pheromones, 
                     int tick_num, int *resource_respawn_timer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Anti-convergence: detect similarity with archetype average
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a.role[i] - (a.arch == i ? 0.7f : 0.1f));
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant role
        int drift_role = (a.arch + 1 + (int)(lcgf(a.rng) * 3)) % 4;
        a.role[drift_role] += lcgf(a.rng) * 0.02f - 0.01f;
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int i = 0; i < 4; i++) a.role[i] /= sum;
    }
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick_num % 50 == 0) {
        float resistance = 1.0f - a.role[3] * 0.5f; // Defenders resist
        a.energy *= 0.5f + 0.5f * resistance;
    }
    
    // Movement based on explore role
    a.vx += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[0];
    a.vy += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[0];
    
    // Velocity damping
    a.vx *= 0.95f;
    a.vy *= 0.95f;
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World boundaries
    if (a.x < 0) { a.x = 0; a.vx = fabsf(a.vx); }
    if (a.x > WORLD_SIZE) { a.x = WORLD_SIZE; a.vx = -fabsf(a.vx); }
    if (a.y < 0) { a.y = 0; a.vy = fabsf(a.vy); }
    if (a.y > WORLD_SIZE) { a.y = WORLD_SIZE; a.vy = -fabsf(a.vy); }
    
    // Check resource collection
    float best_dist = 1.0f;
    int best_res = -1;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = a.x - r.x;
        float dy = a.y - r.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        // Detection range based on explore role
        if (dist < 0.03f + 0.04f * a.role[0]) {
            if (dist < best_dist) {
                best_dist = dist;
                best_res = i;
            }
        }
    }
    
    // Collect resource if in range
    if (best_res != -1) {
        Resource &r = resources[best_res];
        float grab_range = 0.02f + 0.02f * a.role[1];
        
        if (best_dist < grab_range) {
            // Collection bonus based on collect role
            float bonus = 1.0f + 0.5f * a.role[1];
            
            // Territory bonus from nearby defenders of same archetype
            float territory_bonus = 1.0f;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent &other = agents[j];
                if (other.arch != a.arch) continue;
                
                float odx = a.x - other.x;
                float ody = a.y - other.y;
                float odist = sqrtf(odx*odx + ody*ody);
                
                if (odist < 0.08f && other.role[3] > 0.3f) {
                    territory_bonus += 0.2f;
                }
            }
            
            // STIGMERGY: Leave pheromone at resource location
            int px = (int)(r.x * PHEROMONE_GRID);
            int py = (int)(r.y * PHEROMONE_GRID);
            if (px >= 0 && px < PHEROMONE_GRID && py >= 0 && py < PHEROMONE_GRID) {
                int pidx = py * PHEROMONE_GRID + px;
                atomicAdd(&pheromones[pidx].strength[a.arch], 1.0f);
            }
            
            // Energy gain
            a.energy += r.value * bonus * territory_bonus;
            a.fitness += r.value * bonus * territory_bonus;
            r.collected = 1;
            
            // Resource respawn timer
            atomicAdd(&resource_respawn_timer[best_res], 1);
        }
    }
    
    // STIGMERGY: Follow pheromone trails of own archetype
    int px = (int)(a.x * PHEROMONE_GRID);
    int py = (int)(a.y * PHEROMONE_GRID);
    
    if (px > 0 && px < PHEROMONE_GRID-1 && py > 0 && py < PHEROMONE_GRID-1) {
        float max_strength = 0.0f;
        float best_dx = 0.0f, best_dy = 0.0f;
        
        // Check 3x3 neighborhood for strongest pheromone
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                
                int nidx = (py + dy) * PHEROMONE_GRID + (px + dx);
                float strength = pheromones[nidx].strength[a.arch];
                
                if (strength > max_strength) {
                    max_strength = strength;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
        
        // Move toward strongest pheromone (weighted by explore role)
        if (max_strength > 0.1f) {
            a.vx += best_dx * 0.005f * a.role[0];
            a.vy += best_dy * 0.005f * a.role[0];
        }
    }
    
    // Communication: broadcast nearest resource location
    if (a.role[2] > 0.3f && best_res != -1) {
        Resource &r = resources[best_res];
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent &other = agents[j];
            
            float dx = a.x - other.x;
            float dy = a.y - other.y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            // Broadcast to nearby agents of same archetype
            if (dist < 0.06f && other.arch == a.arch) {
                // Attract toward resource
                float attract = 0.01f * a.role[2];
                other.vx += (r.x - other.x) * attract;
                other.vy += (r.y - other.y) * attract;
            }
        }
    }
}

// Respawn resources
__global__ void respawn_resources(Resource *resources, int *resource_respawn_timer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    if (resource_respawn_timer[idx] > 0) {
        resource_respawn_timer[idx]--;
        
        if (resource_respawn_timer[idx] == 0) {
            Resource &r = resources[idx];
            unsigned int rng = idx * 19 + 54321 + clock();
            r.x = (lcg(rng) & 0xFF) / 255.0f;
            r.y = (lcg(rng) & 0xFF) / 255.0f;
            r.value = 0.5f + lcgf(rng) * 0.5f;
            r.collected = 0;
        }
    }
}

int main() {
    printf("Experiment v46: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone trails enhance specialist coordination\n");
    printf("Prediction: Specialists >1.61x advantage over uniform agents\n");
    printf("Baseline: v8 mechanisms (scarcity, territory, comms)\n\n");
    
    // Allocate memory
    Agent *d_agents_spec, *d_agents_uniform;
    Resource *d_resources_spec, *d_resources_uniform;
    Pheromone *d_pheromones_spec, *d_pheromones_uniform;
    int *d_respawn_timer_spec, *d_respawn_timer_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources_spec, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_resources_uniform, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, PHEROMONE_GRID * PHEROMONE_GRID * sizeof(Pheromone));
    cudaMalloc(&d_pheromones_uniform, PHEROMONE_GRID * PHEROMONE_GRID * sizeof(Pheromone));
    cudaMalloc(&d_respawn_timer_spec, RESOURCES * sizeof(int));
    cudaMalloc(&d_respawn_timer_uniform, RESOURCES * sizeof(int));
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    init_pheromones<<<grid_ph, block>>>(d_pheromones_spec);
    init_pheromones<<<grid_ph, block>>>(d_pheromones_uniform);
    
    init_agents<<<grid_spec, block>>>(d_agents_spec, 1);  // Specialized
    init_agents<<<grid_spec, block>>>(d_agents_uniform, 0); // Uniform
    
    init_resources<<<grid_res, block>>>(d_resources_spec);
    init_resources<<<grid_res, block>>>(d_resources_uniform);
    
    cudaMemset(d_respawn_timer_spec, 0, RESOURCES * sizeof(int));
    cudaMemset(d_respawn_timer_uniform, 0, RESOURCES * sizeof(int));
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        // Decay pheromones each tick
        decay_pheromones<<<grid_ph, block>>>(d_pheromones_spec);
        decay_pheromones<<<grid_ph, block>>>(d_pheromones_uniform);
        
        // Run tick for both populations
        tick<<<grid_spec, block>>>(d_agents_spec, d_resources_spec, 
                                  d_pheromones_spec, t, d_respawn_timer_spec);
        tick<<<grid_spec, block>>>(d_agents_uniform, d_resources_uniform, 
                                  d_pheromones_uniform, t, d_respawn_timer_uniform);
        
        // Respawn resources
        respawn_resources<<<grid_res, block>>>(d_resources_spec, d_respawn_timer_spec);
        respawn_resources<<<grid_res, block>>>(d_resources_uniform, d_respawn_timer_uniform);
        
        if (t % 100 == 0) {
            printf("Tick %d/500 completed\n", t);
        }
    }
    
    cudaDeviceS