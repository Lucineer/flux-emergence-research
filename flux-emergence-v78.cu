// CUDA Simulation Experiment v78: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone trails at resource locations that decay over time.
// Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents.
// Expected: Specialists will use pheromones more effectively, leading to >1.61x ratio (v8 baseline).
// Novelty: Stigmergy (indirect communication through environment modification) not tested in previous experiments.

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK = 256;
const float WORLD_SIZE = 1.0f;
const float ENERGY_DECAY = 0.999f;
const float PERTURB_PROB = 0.001f;
const float DRIFT_STRENGTH = 0.01f;
const float SIMILARITY_THRESH = 0.9f;
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;

// Agent archetypes: 0=explorer, 1=collector, 2=communicator, 3=defender
const int ARCH_COUNT = 4;

// Pheromone constants
const int PHEROMONE_GRID = 64; // 64x64 grid
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_STRENGTH = 0.5f;
const float PHEROMONE_SENSE_RANGE = 0.1f;

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCH_COUNT];
    float fitness;
    int arch;
    unsigned int rng;
};

// Resource struct
struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone grid (device global memory)
__device__ float d_pheromone[PHEROMONE_GRID][PHEROMONE_GRID];

// Linear Congruential Generator
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize pheromone grid
__global__ void initPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PHEROMONE_GRID * PHEROMONE_GRID) {
        int i = idx / PHEROMONE_GRID;
        int j = idx % PHEROMONE_GRID;
        d_pheromone[i][j] = 0.0f;
    }
}

// Decay pheromones each tick
__global__ void decayPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PHEROMONE_GRID * PHEROMONE_GRID) {
        int i = idx / PHEROMONE_GRID;
        int j = idx % PHEROMONE_GRID;
        d_pheromone[i][j] *= PHEROMONE_DECAY;
    }
}

// Initialize agents and resources
__global__ void init(Agent *agents, Resource *resources, int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize agents
    if (idx < AGENTS) {
        Agent &a = agents[idx];
        a.rng = 12345 + idx * 6789;
        a.x = lcgf(a.rng) * WORLD_SIZE;
        a.y = lcgf(a.rng) * WORLD_SIZE;
        a.vx = lcgf(a.rng) * 0.02f - 0.01f;
        a.vy = lcgf(a.rng) * 0.02f - 0.01f;
        a.energy = 1.0f;
        a.fitness = 0.0f;
        a.arch = idx % ARCH_COUNT;
        
        if (specialized) {
            // Specialized: strong in one role, weak in others
            for (int i = 0; i < ARCH_COUNT; i++) {
                a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
            }
        } else {
            // Uniform: all roles equal
            for (int i = 0; i < ARCH_COUNT; i++) {
                a.role[i] = 0.25f;
            }
        }
    }
    
    // Initialize resources
    if (idx < RESOURCES) {
        Resource &r = resources[idx];
        unsigned int rng = 54321 + idx * 9876;
        r.x = lcgf(rng) * WORLD_SIZE;
        r.y = lcgf(rng) * WORLD_SIZE;
        r.value = 0.5f + lcgf(rng) * 0.5f;
        r.collected = 0;
    }
}

// Main simulation tick
__global__ void tick(Agent *agents, Resource *resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= ENERGY_DECAY;
    
    // Random perturbation
    if (lcgf(a.rng) < PERTURB_PROB) {
        if (a.arch != 3) { // Defenders resist perturbation
            a.energy *= 0.5f;
        }
        a.vx += lcgf(a.rng) * 0.02f - 0.01f;
        a.vy += lcgf(a.rng) * 0.02f - 0.01f;
    }
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = lcg(a.rng) % AGENTS;
    Agent &other = agents[other_idx];
    if (other_idx != idx) {
        float similarity = 0.0f;
        for (int i = 0; i < ARCH_COUNT; i++) {
            similarity += fabsf(a.role[i] - other.role[i]);
        }
        similarity = 1.0f - similarity / ARCH_COUNT;
        
        if (similarity > SIMILARITY_THRESH) {
            // Apply random drift to non-dominant role
            int drift_role = lcg(a.rng) % ARCH_COUNT;
            while (drift_role == a.arch) drift_role = lcg(a.rng) % ARCH_COUNT;
            a.role[drift_role] += (lcgf(a.rng) * 2.0f - 1.0f) * DRIFT_STRENGTH;
            a.role[drift_role] = fmaxf(0.0f, fminf(1.0f, a.role[drift_role]));
        }
    }
    
    // Role coupling
    for (int i = 0; i < AGENTS; i += 32) { // Sample 32 others
        int j = (idx + i) % AGENTS;
        if (j == idx) continue;
        Agent &b = agents[j];
        float coupling = (a.arch == b.arch) ? COUPLING_SAME : COUPLING_DIFF;
        for (int r = 0; r < ARCH_COUNT; r++) {
            a.role[r] += (b.role[r] - a.role[r]) * coupling;
            a.role[r] = fmaxf(0.0f, fminf(1.0f, a.role[r]));
        }
    }
    
    // Sense pheromones in vicinity
    float pheromone_x = 0.0f, pheromone_y = 0.0f;
    int grid_x = (int)(a.x * PHEROMONE_GRID);
    int grid_y = (int)(a.y * PHEROMONE_GRID);
    int range = (int)(PHEROMONE_SENSE_RANGE * PHEROMONE_GRID);
    
    for (int dx = -range; dx <= range; dx++) {
        for (int dy = -range; dy <= range; dy++) {
            int px = (grid_x + dx + PHEROMONE_GRID) % PHEROMONE_GRID;
            int py = (grid_y + dy + PHEROMONE_GRID) % PHEROMONE_GRID;
            float dist = sqrtf(dx*dx + dy*dy) / PHEROMONE_GRID;
            if (dist < PHEROMONE_SENSE_RANGE) {
                float strength = d_pheromone[px][py] * (1.0f - dist/PHEROMONE_SENSE_RANGE);
                pheromone_x += dx * strength;
                pheromone_y += dy * strength;
            }
        }
    }
    
    // Movement based on role and pheromones
    float move_x = 0.0f, move_y = 0.0f;
    
    // Explorer: move randomly but follow pheromones weakly
    move_x += a.role[0] * (lcgf(a.rng) * 0.02f - 0.01f);
    move_y += a.role[0] * (lcgf(a.rng) * 0.02f - 0.01f);
    
    // Collector: move toward pheromones strongly
    move_x += a.role[1] * pheromone_x * 0.1f;
    move_y += a.role[1] * pheromone_y * 0.1f;
    
    // Defender: move against pheromones (protect territory)
    move_x -= a.role[3] * pheromone_x * 0.05f;
    move_y -= a.role[3] * pheromone_y * 0.05f;
    
    // Apply movement
    a.vx = a.vx * 0.9f + move_x * 0.1f;
    a.vy = a.vy * 0.9f + move_y * 0.1f;
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.02f) {
        a.vx *= 0.02f / speed;
        a.vy *= 0.02f / speed;
    }
    
    a.x += a.vx;
    a.y += a.vy;
    
    // Wrap around world
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
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        // Wrap distance
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
        
        // Collector grab range with bonus
        float grab_range = 0.02f + a.role[1] * 0.02f;
        if (dist < grab_range) {
            float bonus = 1.0f + a.role[1] * 0.5f; // Up to 50% bonus
            
            // Defender territory boost
            int defender_count = 0;
            for (int j = 0; j < AGENTS; j += 16) {
                Agent &d = agents[(idx + j) % AGENTS];
                if (d.arch == 3) {
                    float ddx = d.x - a.x;
                    float ddy = d.y - a.y;
                    if (ddx > 0.5f * WORLD_SIZE) ddx -= WORLD_SIZE;
                    if (ddx < -0.5f * WORLD_SIZE) ddx += WORLD_SIZE;
                    if (ddy > 0.5f * WORLD_SIZE) ddy -= WORLD_SIZE;
                    if (ddy < -0.5f * WORLD_SIZE) ddy += WORLD_SIZE;
                    if (sqrtf(ddx*ddx + ddy*ddy) < 0.1f) defender_count++;
                }
            }
            bonus += defender_count * 0.2f; // 20% per defender
            
            float gain = r.value * bonus;
            a.energy += gain;
            a.fitness += gain;
            r.collected = 1;
            
            // Leave pheromone at resource location
            int px = (int)(r.x * PHEROMONE_GRID);
            int py = (int)(r.y * PHEROMONE_GRID);
            atomicAdd(&d_pheromone[px][py], PHEROMONE_STRENGTH);
            
            break;
        }
    }
    
    // Communicator broadcast
    if (a.role[2] > 0.3f && best_res != -1) {
        Resource &r = resources[best_res];
        float comm_range = 0.06f;
        for (int i = 0; i < AGENTS; i += 8) {
            Agent &b = agents[(idx + i) % AGENTS];
            if (&b == &a) continue;
            
            float dx = b.x - a.x;
            float dy = b.y - a.y;
            if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
            if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
            if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
            if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
            
            if (sqrtf(dx*dx + dy*dy) < comm_range) {
                // Influence neighbor's movement toward resource
                float influence = a.role[2] * 0.1f;
                b.vx += (r.x - b.x) * influence;
                b.vy += (r.y - b.y) * influence;
            }
        }
    }
}

int main() {
    printf("Experiment v78: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone trails enhance specialist coordination\n");
    printf("Prediction: Specialist advantage >1.61x (v8 baseline)\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RESOURCES, TICKS);
    
    // Allocate memory
    Agent *d_agents_spec, *d_agents_unif;
    Resource *d_res_spec, *d_res_unif;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_unif, AGENTS * sizeof(Agent));
    cudaMalloc(&d_res_spec, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_res_unif, RESOURCES * sizeof(Resource));
    
    // Initialize pheromone grid
    initPheromone<<<(PHEROMONE_GRID*PHEROMONE_GRID + BLOCK - 1)/BLOCK, BLOCK>>>();
    cudaDeviceSynchronize();
    
    // Initialize populations
    init<<<(AGENTS + BLOCK - 1)/BLOCK, BLOCK>>>(d_agents_spec, d_res_spec, 1);
    init<<<(AGENTS + BLOCK - 1)/BLOCK, BLOCK>>>(d_agents_unif, d_res_unif, 0);
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        // Decay pheromones each tick
        decayPheromone<<<(PHEROMONE_GRID*PHEROMONE_GRID + BLOCK - 1)/BLOCK, BLOCK>>>();
        
        // Run specialized population
        tick<<<(AGENTS + BLOCK - 1)/BLOCK, BLOCK>>>(d_agents_spec, d_res_spec, t);
        
        // Reset resources for uniform population (separate environment)
        if (t % 50 == 49) {
            cudaMemset(d_res_unif, 0, RESOURCES * sizeof(Resource));
            init<<<(RESOURCES + BLOCK - 1)/BLOCK, BLOCK>>>(d_agents_unif, d_res_unif, 0);
        }
        
        // Run uniform population
        tick<<<(AGENTS + BLOCK - 1)/BLOCK, BLOCK>>>(d_agents_unif, d_res_unif, t);
        
        cudaDeviceSynchronize();
        
        // Progress indicator
        if (t % 100 == 99) printf("Tick %d/500\n", t + 1);
   