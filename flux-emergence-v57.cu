
/*
CUDA Simulation Experiment v57: STIGMERY TRAILS
Testing: Agents leave pheromone trails at resource collection sites that decay over time.
Prediction: Pheromones will create positive feedback loops, allowing specialized agents
to form efficient foraging trails, increasing their advantage over uniform agents.
Baseline: Includes v8 confirmed mechanisms (scarcity, territory, comms).
Comparison: Specialized (role[arch]=0.7) vs Uniform (all roles=0.25).
Expected: Specialists should show >1.61x advantage due to trail optimization.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define AGENTS 1024
#define RESOURCES 128
#define TICKS 500
#define ARCHETYPES 4
#define THREADS 256
#define PHEROMONE_GRID 128

// Linear Congruential Generator
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

struct Resource {
    float x, y;
    float value;
    bool collected;
    float pheromone; // v57: pheromone deposited when collected
};

struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPES];
    float fitness;
    int arch;
    unsigned int rng;
    float memory_x, memory_y; // Remembered resource location
};

// v57: Pheromone grid for stigmergy
__device__ float pheromone_grid[PHEROMONE_GRID][PHEROMONE_GRID];

__global__ void init_agents(Agent* agents, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    a.rng = seed + idx * 17;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = lcgf(a.rng) * 0.02f - 0.01f;
    a.vy = lcgf(a.rng) * 0.02f - 0.01f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.memory_x = -1.0f;
    a.memory_y = -1.0f;
    
    // Specialized vs Uniform population split
    if (idx < AGENTS/2) {
        // Specialized: role[arch]=0.7, others 0.1
        a.arch = idx % ARCHETYPES;
        for (int i = 0; i < ARCHETYPES; i++) {
            a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform: all roles 0.25
        a.arch = idx % ARCHETYPES;
        for (int i = 0; i < ARCHETYPES; i++) {
            a.role[i] = 0.25f;
        }
    }
}

__global__ void init_resources(Resource* resources, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource& r = resources[idx];
    unsigned int rng = seed + idx * 29;
    r.x = lcgf(rng);
    r.y = lcgf(rng);
    r.value = 0.8f + lcgf(rng) * 0.4f;
    r.collected = false;
    r.pheromone = 0.0f;
}

__global__ void init_pheromone_grid() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < PHEROMONE_GRID && y < PHEROMONE_GRID) {
        pheromone_grid[x][y] = 0.0f;
    }
}

__device__ float role_similarity(const Agent& a, const Agent& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) {
        dot += a.role[i] * b.role[i];
        norm_a += a.role[i] * a.role[i];
        norm_b += b.role[i] * b.role[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

__device__ void anti_convergence(Agent& a, unsigned int& rng) {
    // Find dominant role
    int dominant = 0;
    float max_role = a.role[0];
    for (int i = 1; i < ARCHETYPES; i++) {
        if (a.role[i] > max_role) {
            max_role = a.role[i];
            dominant = i;
        }
    }
    
    // Check if too similar to uniform
    float uniformity = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) {
        uniformity += fabsf(a.role[i] - 0.25f);
    }
    if (uniformity < 0.1f) return;
    
    // Apply drift to non-dominant roles
    for (int i = 0; i < ARCHETYPES; i++) {
        if (i != dominant) {
            a.role[i] += (lcgf(rng) - 0.5f) * 0.01f;
            a.role[i] = fmaxf(0.05f, fminf(0.7f, a.role[i]));
        }
    }
    
    // Renormalize
    float sum = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) sum += a.role[i];
    for (int i = 0; i < ARCHETYPES; i++) a.role[i] /= sum;
}

__device__ float get_pheromone(float x, float y) {
    int gx = (int)(x * PHEROMONE_GRID);
    int gy = (int)(y * PHEROMONE_GRID);
    gx = max(0, min(PHEROMONE_GRID-1, gx));
    gy = max(0, min(PHEROMONE_GRID-1, gy));
    return pheromone_grid[gx][gy];
}

__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Random perturbation (defenders resist)
    if (lcgf(a.rng) < 0.01f) {
        float resistance = 1.0f - a.role[3] * 0.5f; // defenders resist
        if (lcgf(a.rng) > resistance) {
            a.vx += (lcgf(a.rng) - 0.5f) * 0.05f;
            a.vy += (lcgf(a.rng) - 0.5f) * 0.05f;
            a.energy *= 0.5f;
        }
    }
    
    // Movement with pheromone attraction (v57)
    float px = a.x + a.vx;
    float py = a.y + a.vy;
    
    // Sample pheromone in 3x3 grid around potential new position
    float pheromone_attraction = 0.0f;
    float best_px = px, best_py = py;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            float sample_x = fmodf(px + dx*0.02f + 1.0f, 1.0f);
            float sample_y = fmodf(py + dy*0.02f + 1.0f, 1.0f);
            float p = get_pheromone(sample_x, sample_y);
            if (p > pheromone_attraction) {
                pheromone_attraction = p;
                best_px = sample_x;
                best_py = sample_y;
            }
        }
    }
    
    // Blend movement with pheromone attraction
    if (pheromone_attraction > 0.1f && lcgf(a.rng) < a.role[0]) { // explorers use pheromones
        a.x = best_px;
        a.y = best_py;
    } else {
        a.x = fmodf(px + 1.0f, 1.0f);
        a.y = fmodf(py + 1.0f, 1.0f);
    }
    
    // Keep velocity within bounds
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.03f) {
        a.vx *= 0.03f / speed;
        a.vy *= 0.03f / speed;
    }
    
    // Interaction with resources
    for (int r = 0; r < RESOURCES; r++) {
        Resource& res = resources[r];
        if (res.collected) continue;
        
        float dx = a.x - res.x;
        float dy = a.y - res.y;
        dx -= roundf(dx);
        dy -= roundf(dy);
        float dist = sqrtf(dx*dx + dy*dy);
        
        // Detection (explorers)
        if (dist < 0.03f + a.role[0]*0.04f) {
            a.memory_x = res.x;
            a.memory_y = res.y;
        }
        
        // Collection (collectors)
        if (dist < 0.02f + a.role[1]*0.02f) {
            float bonus = 1.0f + a.role[1]*0.5f;
            a.energy += res.value * bonus;
            a.fitness += res.value * bonus;
            res.collected = true;
            
            // v57: Deposit pheromone at collection site
            int gx = (int)(res.x * PHEROMONE_GRID);
            int gy = (int)(res.y * PHEROMONE_GRID);
            gx = max(0, min(PHEROMONE_GRID-1, gx));
            gy = max(0, min(PHEROMONE_GRID-1, gy));
            atomicAdd(&pheromone_grid[gx][gy], 0.5f);
            res.pheromone = 0.5f;
        }
    }
    
    // Communication (communicators)
    if (a.memory_x >= 0 && lcgf(a.rng) < a.role[2]*0.1f) {
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent& other = agents[i];
            float dx = a.x - other.x;
            float dy = a.y - other.y;
            dx -= roundf(dx);
            dy -= roundf(dy);
            if (dx*dx + dy*dy < 0.06f*0.06f) {
                other.memory_x = a.memory_x;
                other.memory_y = a.memory_y;
            }
        }
    }
    
    // Territory defense bonus
    int defenders_nearby = 0;
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent& other = agents[i];
        if (other.arch != a.arch) continue;
        float dx = a.x - other.x;
        float dy = a.y - other.y;
        dx -= roundf(dx);
        dy -= roundf(dy);
        if (dx*dx + dy*dy < 0.04f*0.04f && other.role[3] > 0.3f) {
            defenders_nearby++;
        }
    }
    if (defenders_nearby > 0 && a.role[3] > 0.3f) {
        a.energy *= 1.0f + defenders_nearby * 0.2f;
    }
    
    // Social learning with coupling
    if (lcgf(a.rng) < 0.02f) {
        int partner = lcgf(a.rng) * AGENTS;
        Agent& other = agents[partner];
        float sim = role_similarity(a, other);
        float coupling = (a.arch == other.arch) ? 0.02f : 0.002f;
        
        for (int i = 0; i < ARCHETYPES; i++) {
            a.role[i] += (other.role[i] - a.role[i]) * coupling * sim;
        }
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < ARCHETYPES; i++) sum += a.role[i];
        for (int i = 0; i < ARCHETYPES; i++) a.role[i] /= sum;
        
        // Anti-convergence
        if (sim > 0.9f) {
            anti_convergence(a, a.rng);
        }
    }
    
    // Energy-based fitness
    if (a.energy > 2.0f) {
        a.fitness += a.energy - 2.0f;
        a.energy = 2.0f;
    }
    if (a.energy < 0.01f) {
        a.energy = 0.5f;
        a.x = lcgf(a.rng);
        a.y = lcgf(a.rng);
    }
}

__global__ void decay_pheromones() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < PHEROMONE_GRID && y < PHEROMONE_GRID) {
        pheromone_grid[x][y] *= 0.95f; // 5% decay per tick
    }
}

__global__ void respawn_resources(Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource& r = resources[idx];
    if (r.collected && (tick_num % 50 == 0)) {
        unsigned int rng = tick_num * 137 + idx * 29;
        r.x = lcgf(rng);
        r.y = lcgf(rng);
        r.value = 0.8f + lcgf(rng) * 0.4f;
        r.collected = false;
        r.pheromone = 0.0f;
    }
}

int main() {
    // Allocate memory
    Agent* agents;
    Resource* resources;
    cudaMallocManaged(&agents, AGENTS * sizeof(Agent));
    cudaMallocManaged(&resources, RESOURCES * sizeof(Resource));
    
    // Initialize
    dim3 block(THREADS);
    dim3 grid_agents((AGENTS + THREADS - 1) / THREADS);
    dim3 grid_res((RESOURCES + THREADS - 1) / THREADS);
    
    init_agents<<<grid_agents, block>>>(agents, 12345);
    init_resources<<<grid_res, block>>>(resources, 67890);
    
    dim3 pheromone_threads(16, 16);
    dim3 pheromone_blocks((PHEROMONE_GRID + 15) / 16, (PHEROMONE_GRID + 15) / 16);
    init_pheromone_grid<<<pheromone_blocks, pheromone_threads>>>();
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        tick<<<grid_agents, block>>>(agents, resources, t);
        decay_pheromones<<<pheromone_blocks, pheromone_threads>>>();
        if (t % 50 == 49) {
            respawn_resources<<<grid_res, block>>>(resources, t);
        }
        cudaDeviceSynchronize();
    }
    
    // Calculate results
    float spec_fitness = 0.0f, unif_fitness = 0.0f;
    float spec_roles[ARCHETYPES] = {0};
    float unif_roles[ARCHETYPES] = {0};
    
    for (int i = 0; i < AGENTS; i++) {
        if (i < AGENTS/2) {
            spec_fitness += agents[i].fitness;
            for (int j = 0; j < ARCHETYPES; j++) {
                spec_roles[j] += agents[i].role[j];
            }
        } else {
            unif_fitness += agents[i].fitness;
            for (int j = 0; j < ARCHETYPES; j++) {
                unif_roles[j] += agents[i].role[j];
            }
        }
    }
    
    spec_fitness /= (AGENTS/2);
    unif_fitness /= (AGENTS/2);
    for (int j = 0; j < ARCHETYPES; j++) {
        spec_