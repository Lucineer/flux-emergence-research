
/*
CUDA Simulation Experiment v69: STIGMERGY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents (ratio > 1.61x from v8 baseline).
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence)
Novel: Stigmergy - agents deposit pheromone when collecting resources, others follow trails
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 256; // 256x256 grid for pheromone map
const float WORLD_SIZE = 1.0f;

// Agent archetypes
enum { ARCH_GENERALIST = 0, ARCH_SPECIALIST = 1 };

// Agent structure
struct Agent {
    float x, y;           // position
    float vx, vy;         // velocity
    float energy;         // energy level
    float role[4];        // explore, collect, communicate, defend
    float fitness;        // accumulated fitness
    int arch;             // archetype
    unsigned int rng;     // random state
};

// Resource structure
struct Resource {
    float x, y;           // position
    float value;          // resource value
    int collected;        // collection flag
};

// Pheromone structure for stigmergy
struct Pheromone {
    float strength[2];    // separate pheromone for each archetype
};

// Linear congruential generator (device/host)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, Pheromone* pheromone) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    a->rng = idx * 17 + 12345;
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->vy = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    // Assign archetype: half specialists, half generalists
    a->arch = (idx < AGENTS/2) ? ARCH_SPECIALIST : ARCH_GENERALIST;
    
    // Role initialization
    if (a->arch == ARCH_SPECIALIST) {
        // Specialists: biased toward one role based on thread index
        int primary_role = idx % 4;
        for (int i = 0; i < 4; i++) {
            a->role[i] = (i == primary_role) ? 0.7f : 0.1f;
        }
    } else {
        // Generalists: uniform roles
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    unsigned int rng = idx * 19 + 54321;
    r->x = (lcg(&rng) % 10000) / 10000.0f;
    r->y = (lcg(&rng) % 10000) / 10000.0f;
    r->value = 0.5f + (lcg(&rng) % 1000) / 2000.0f;
    r->collected = 0;
}

// Initialize pheromone grid kernel
__global__ void init_pheromone(Pheromone* pheromone) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    pheromone[idx].strength[0] = 0.0f;
    pheromone[idx].strength[1] = 0.0f;
}

// Decay pheromone kernel (called each tick)
__global__ void decay_pheromone(Pheromone* pheromone) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    // Exponential decay: 5% per tick
    pheromone[idx].strength[0] *= 0.95f;
    pheromone[idx].strength[1] *= 0.95f;
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromone, int tick_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with nearby agents
    int similar_count = 0;
    int total_count = 0;
    
    // Movement with pheromone following
    float fx = 0.0f, fy = 0.0f;
    
    // Sample pheromone in neighborhood
    int grid_x = (int)(a->x * PHEROMONE_GRID);
    int grid_y = (int)(a->y * PHEROMONE_GRID);
    
    // Check 3x3 neighborhood for pheromone gradient
    float max_ph = 0.0f;
    int best_dx = 0, best_dy = 0;
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int px = grid_x + dx;
            int py = grid_y + dy;
            if (px >= 0 && px < PHEROMONE_GRID && py >= 0 && py < PHEROMONE_GRID) {
                int pidx = py * PHEROMONE_GRID + px;
                float ph_strength = pheromone[pidx].strength[a->arch];
                if (ph_strength > max_ph) {
                    max_ph = ph_strength;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
    }
    
    // Follow pheromone gradient if strong enough
    if (max_ph > 0.1f) {
        fx += best_dx * 0.01f * a->role[0]; // explore role affects following strength
        fy += best_dy * 0.01f * a->role[0];
    }
    
    // Random movement component
    fx += (lcgf(&a->rng) - 0.5f) * 0.02f;
    fy += (lcgf(&a->rng) - 0.5f) * 0.02f;
    
    // Update velocity and position
    a->vx = a->vx * 0.9f + fx * 0.1f;
    a->vy = a->vy * 0.9f + fy * 0.1f;
    
    a->x += a->vx;
    a->y += a->vy;
    
    // World boundaries (bouncing)
    if (a->x < 0) { a->x = 0; a->vx = fabs(a->vx); }
    if (a->x >= WORLD_SIZE) { a->x = WORLD_SIZE - 0.0001f; a->vx = -fabs(a->vx); }
    if (a->y < 0) { a->y = 0; a->vy = fabs(a->vy); }
    if (a->y >= WORLD_SIZE) { a->y = WORLD_SIZE - 0.0001f; a->vy = -fabs(a->vy); }
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    // Explore role: detection range 0.03-0.07
    float detect_range = 0.03f + a->role[0] * 0.04f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    // Collect resource if in range
    if (best_res != -1) {
        Resource* r = &resources[best_res];
        float grab_range = 0.02f + a->role[1] * 0.02f;
        
        if (best_dist < grab_range) {
            // Collection bonus based on collect role
            float bonus = 1.0f + a->role[1] * 0.5f;
            float gain = r->value * bonus;
            
            // Territory bonus: check for nearby defenders of same archetype
            float territory_bonus = 1.0f;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent* other = &agents[j];
                if (other->arch != a->arch) continue;
                
                float odx = other->x - a->x;
                float ody = other->y - a->y;
                float odist = sqrtf(odx*odx + ody*ody);
                
                if (odist < 0.05f && other->role[3] > 0.5f) {
                    territory_bonus += 0.2f;
                }
            }
            
            gain *= territory_bonus;
            a->energy += gain;
            a->fitness += gain;
            
            // STIGMERGY: Deposit pheromone at resource location
            int ph_x = (int)(r->x * PHEROMONE_GRID);
            int ph_y = (int)(r->y * PHEROMONE_GRID);
            if (ph_x >= 0 && ph_x < PHEROMONE_GRID && ph_y >= 0 && ph_y < PHEROMONE_GRID) {
                int ph_idx = ph_y * PHEROMONE_GRID + ph_x;
                atomicAdd(&pheromone[ph_idx].strength[a->arch], 0.5f);
            }
            
            r->collected = 1;
        }
    }
    
    // Communication role: broadcast resource locations
    if (a->role[2] > 0.3f && best_res != -1) {
        Resource* r = &resources[best_res];
        float comm_range = 0.06f;
        
        // In a real implementation, this would communicate to nearby agents
        // For simplicity, we just deposit pheromone at the resource location
        int ph_x = (int)(r->x * PHEROMONE_GRID);
        int ph_y = (int)(r->y * PHEROMONE_GRID);
        if (ph_x >= 0 && ph_x < PHEROMONE_GRID && ph_y >= 0 && ph_y < PHEROMONE_GRID) {
            int ph_idx = ph_y * PHEROMONE_GRID + ph_x;
            atomicAdd(&pheromone[ph_idx].strength[a->arch], 0.3f * a->role[2]);
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_id % 50 == 0 && lcgf(&a->rng) < 0.1f) {
        // Defenders resist perturbation
        if (a->role[3] < 0.5f) {
            a->energy *= 0.5f;
        } else {
            a->energy *= 0.8f; // partial resistance
        }
    }
    
    // Anti-convergence: random drift when too similar
    if (tick_id % 100 == 0) {
        // Find dominant role
        int dominant = 0;
        for (int i = 1; i < 4; i++) {
            if (a->role[i] > a->role[dominant]) dominant = i;
        }
        
        // Apply small random drift to non-dominant roles
        for (int i = 0; i < 4; i++) {
            if (i != dominant) {
                a->role[i] += (lcgf(&a->rng) - 0.5f) * 0.01f;
                a->role[i] = fmaxf(0.0f, fminf(1.0f, a->role[i]));
            }
        }
        
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) {
            a->role[i] /= sum;
        }
    }
}

// Reset resources kernel
__global__ void reset_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    if (r->collected) {
        unsigned int rng = idx * 19 + 54321 + 123;
        r->x = (lcg(&rng) % 10000) / 10000.0f;
        r->y = (lcg(&rng) % 10000) / 10000.0f;
        r->value = 0.5f + (lcg(&rng) % 1000) / 2000.0f;
        r->collected = 0;
    }
}

int main() {
    // Allocate memory
    Agent* d_agents;
    Resource* d_resources;
    Pheromone* d_pheromone;
    
    cudaMalloc(&d_agents, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromone, PHEROMONE_GRID * PHEROMONE_GRID * sizeof(Pheromone));
    
    // Initialize
    dim3 block(256);
    dim3 grid_agents((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    init_pheromone<<<grid_ph, block>>>(d_pheromone);
    init_agents<<<grid_agents, block>>>(d_agents, d_pheromone);
    init_resources<<<grid_res, block>>>(d_resources);
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int tick_id = 0; tick_id < TICKS; tick_id++) {
        // Decay pheromone each tick
        decay_pheromone<<<grid_ph, block>>>(d_pheromone);
        
        // Main simulation tick
        tick<<<grid_agents, block>>>(d_agents, d_resources, d_pheromone, tick_id);
        cudaDeviceSynchronize();
        
        // Reset resources periodically
        if (tick_id % 50 == 49) {
            reset_resources<<<grid_res, block>>>(d_resources);
            cudaDeviceSynchronize();
        }
    }
    
    // Copy results back
    Agent* agents = new Agent[AGENTS];
    cudaMemcpy(agents, d_agents, AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float specialist_fitness = 0.0f;
    float generalist_fitness = 0.0f;
    int specialist_count = 0;
    int generalist_count = 0;
    
    for (int i = 0; i < AGENTS; i++) {
        if (agents[i].arch == ARCH_SPECIALIST) {
            specialist_fitness += agents[i].fitness;
            specialist_count++;
        } else {
            generalist_fitness += agents[i].fitness;
            generalist_count++;
        }
    }
    
    specialist_fitness /= specialist_count;
    generalist_fitness /= generalist_count;
    float ratio = specialist_fitness / generalist_fitness;
    
    // Print results
    printf("=== EXPERIMENT v69: STIGMERGY TRAILS ===\n");
    printf("Configuration:\n");
    printf("  Agents: %d (512 specialists, 512 generalists)\n", AGENTS);
    printf("  Resources: %d (scarce)\n", RESOURCES);
    printf("  Ticks: %d\n", TICKS