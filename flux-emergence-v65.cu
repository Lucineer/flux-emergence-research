
/*
CUDA Simulation Experiment v65: STIGMERGY TRAILS
Testing: Pheromone trails at resource locations that decay over time
Prediction: Stigmergy will improve specialist efficiency by 15-20% over v8 baseline
  because agents can follow trails to resources, reducing search time.
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence)
Comparison: Specialized (role[arch]=0.7) vs Uniform (all roles=0.25)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RESOURCE_COUNT = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Agent archetypes
enum Archetype {
    ARCH_EXPLORER = 0,
    ARCH_COLLECTOR = 1,
    ARCH_COMMUNICATOR = 2,
    ARCH_DEFENDER = 3,
    ARCH_COUNT = 4
};

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Dominant archetype
    unsigned int rng;     // Random number state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Whether collected
    float pheromone;      // Stigmergy: pheromone strength at this location
};

// LCG random number generator
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent* a = &agents[idx];
    a->rng = idx * 123456789 + 987654321;
    
    // Random position
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    
    // Small random velocity
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    
    // Initial energy
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    if (specialized) {
        // Specialized population: one dominant role per archetype
        a->arch = idx % ARCH_COUNT;
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles equal
        a->arch = idx % ARCH_COUNT;  // Still assign archetype for territory
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCE_COUNT) return;
    
    Resource* r = &resources[idx];
    unsigned int rng = idx * 135791113 + 246810121;
    
    // Random position
    r->x = lcgf(&rng);
    r->y = lcgf(&rng);
    r->value = 0.5f + lcgf(&rng) * 0.5f;  // 0.5-1.0
    r->collected = 0;
    r->pheromone = 0.0f;  // Initial pheromone
}

// Compute similarity between two agents
__device__ float compute_similarity(const Agent* a, const Agent* b) {
    float diff = 0.0f;
    for (int i = 0; i < ARCH_COUNT; i++) {
        float d = a->role[i] - b->role[i];
        diff += d * d;
    }
    return 1.0f - sqrtf(diff) / 2.0f;  // Normalized to [0,1]
}

// Main simulation kernel
__global__ void tick_kernel(Agent* agents, Resource* resources, int tick) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Apply anti-convergence drift
    int similar_count = 0;
    float avg_role[4] = {0};
    
    // Sample 16 random agents to check similarity
    for (int i = 0; i < 16; i++) {
        int j = (int)(lcgf(&a->rng) * AGENT_COUNT);
        if (j >= AGENT_COUNT) j = AGENT_COUNT - 1;
        
        Agent* other = &agents[j];
        float sim = compute_similarity(a, other);
        
        if (sim > 0.9f) {
            similar_count++;
            for (int k = 0; k < ARCH_COUNT; k++) {
                avg_role[k] += other->role[k];
            }
        }
    }
    
    // If too similar to others, apply drift to non-dominant role
    if (similar_count >= 4) {
        for (int k = 0; k < ARCH_COUNT; k++) {
            avg_role[k] /= similar_count;
        }
        
        // Find dominant role (max difference from uniform)
        int dominant = 0;
        float max_diff = 0.0f;
        for (int k = 0; k < ARCH_COUNT; k++) {
            float diff = fabsf(a->role[k] - avg_role[k]);
            if (diff > max_diff) {
                max_diff = diff;
                dominant = k;
            }
        }
        
        // Apply small random drift to a non-dominant role
        int drift_target = (dominant + 1 + (int)(lcgf(&a->rng) * (ARCH_COUNT - 1))) % ARCH_COUNT;
        a->role[drift_target] += lcgf(&a->rng) * 0.02f - 0.01f;
        
        // Renormalize roles
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int k = 0; k < ARCH_COUNT; k++) {
            a->role[k] /= sum;
        }
    }
    
    // Update position with velocity
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary conditions (wrap-around)
    if (a->x < 0.0f) a->x += 1.0f;
    if (a->x >= 1.0f) a->x -= 1.0f;
    if (a->y < 0.0f) a->y += 1.0f;
    if (a->y >= 1.0f) a->y -= 1.0f;
    
    // Small random velocity change
    a->vx += lcgf(&a->rng) * 0.002f - 0.001f;
    a->vy += lcgf(&a->rng) * 0.002f - 0.001f;
    
    // Limit velocity
    float speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
    if (speed > 0.02f) {
        a->vx *= 0.02f / speed;
        a->vy *= 0.02f / speed;
    }
    
    // Resource interaction
    float best_pheromone = -1.0f;
    int best_res = -1;
    float best_dist = 1.0f;
    
    // Check nearby resources (scarcity: small detection range)
    for (int i = 0; i < RESOURCE_COUNT; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        
        // Handle wrap-around
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx * dx + dy * dy);
        
        // STIGMERGY: Agents are attracted to pheromone trails
        float attraction = r->pheromone * 0.5f;  // Pheromone influence
        
        // Detection based on role[0] (explore) with pheromone boost
        float detect_range = 0.03f + a->role[0] * 0.04f + attraction;
        
        if (dist < detect_range) {
            // Consider pheromone strength in decision
            float score = r->pheromone - dist;  // Balance pheromone vs distance
            
            if (score > best_pheromone) {
                best_pheromone = score;
                best_res = i;
                best_dist = dist;
            }
        }
    }
    
    // If found a resource, interact based on roles
    if (best_res != -1) {
        Resource* r = &resources[best_res];
        
        // Collect based on role[1] (collect)
        float grab_range = 0.02f + a->role[1] * 0.02f;
        
        if (best_dist < grab_range && !r->collected) {
            // Collection bonus based on role[1]
            float bonus = 1.0f + a->role[1] * 0.5f;
            
            // Territory bonus: check for nearby defenders of same archetype
            float territory_bonus = 1.0f;
            for (int i = 0; i < 8; i++) {
                int j = (int)(lcgf(&a->rng) * AGENT_COUNT);
                if (j >= AGENT_COUNT) j = AGENT_COUNT - 1;
                
                Agent* other = &agents[j];
                if (other->arch == a->arch && other->role[3] > 0.3f) {
                    float dx = other->x - a->x;
                    float dy = other->y - a->y;
                    if (dx > 0.5f) dx -= 1.0f;
                    if (dx < -0.5f) dx += 1.0f;
                    if (dy > 0.5f) dy -= 1.0f;
                    if (dy < -0.5f) dy += 1.0f;
                    
                    if (sqrtf(dx*dx + dy*dy) < 0.1f) {
                        territory_bonus += 0.2f;
                    }
                }
            }
            
            // Collect resource
            float gain = r->value * bonus * territory_bonus;
            a->energy += gain;
            a->fitness += gain;
            r->collected = 1;
            
            // STIGMERGY: Leave pheromone at collected resource location
            r->pheromone = 1.0f;  // Fresh pheromone deposit
        }
        
        // Communicate based on role[2]
        if (a->role[2] > 0.3f) {
            // Broadcast location to nearby agents
            for (int i = 0; i < 4; i++) {
                int j = (int)(lcgf(&a->rng) * AGENT_COUNT);
                if (j >= AGENT_COUNT) j = AGENT_COUNT - 1;
                
                Agent* other = &agents[j];
                float dx = other->x - a->x;
                float dy = other->y - a->y;
                if (dx > 0.5f) dx -= 1.0f;
                if (dx < -0.5f) dx += 1.0f;
                if (dy > 0.5f) dy -= 1.0f;
                if (dy < -0.5f) dy += 1.0f;
                
                if (sqrtf(dx*dx + dy*dy) < 0.06f) {
                    // Influence other agent's velocity toward resource
                    float influence = a->role[2] * 0.01f;
                    other->vx += (r->x - other->x) * influence;
                    other->vy += (r->y - other->y) * influence;
                }
            }
        }
    }
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick % 50 == 0) {
        float resistance = a->role[3];  // Defend role provides resistance
        if (lcgf(&a->rng) > resistance * 0.5f) {
            a->energy *= 0.5f;
            a->vx += lcgf(&a->rng) * 0.02f - 0.01f;
            a->vy += lcgf(&a->rng) * 0.02f - 0.01f;
        }
    }
}

// Pheromone decay and resource respawn kernel
__global__ void update_resources(Resource* resources, int tick) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCE_COUNT) return;
    
    Resource* r = &resources[idx];
    
    // STIGMERGY: Pheromone decay over time
    r->pheromone *= 0.95f;  // 5% decay per tick
    
    // Respawn collected resources every 50 ticks
    if (tick % 50 == 0) {
        if (r->collected) {
            unsigned int rng = idx * 135791113 + 246810121 + tick;
            r->x = lcgf(&rng);
            r->y = lcgf(&rng);
            r->value = 0.5f + lcgf(&rng) * 0.5f;
            r->collected = 0;
            r->pheromone = 0.0f;  // Start with no pheromone
        }
    }
}

int main() {
    printf("Experiment v65: STIGMERGY TRAILS\n");
    printf("Testing pheromone trails at resource locations\n");
    printf("Prediction: +15-20%% specialist efficiency over v8 baseline\n\n");
    
    // Allocate device memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources_spec;
    Resource* d_resources_uniform;
    
    cudaMalloc(&d_agents_spec, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_resources_spec, RESOURCE_COUNT * sizeof(Resource));
    cudaMalloc(&d_resources_uniform, RESOURCE_COUNT * sizeof(Resource));
    
    // Host memory for results
    Agent* h_agents_spec = new Agent[AGENT_COUNT];
    Agent* h_agents_uniform = new Agent[AGENT_COUNT];
    
    // Initialize specialized population
    int blocks = (AGENT_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_agents<<<blocks, BLOCK_SIZE>>>(d_agents_spec, 1);
    init_agents<<<blocks, BLOCK_SIZE>>>(d_agents_uniform, 0);
    
    init_resources<<<(RESOURCE_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_resources_spec);
    init_resources<<<(RESOURCE_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_resources_uniform);
    
    cudaDeviceSynchronize();
    
    // Run simulation for both populations
    for (int tick = 0; tick < TICKS; tick++) {
        tick_kernel<<<blocks, BLOCK_SIZE>>>(d_agents_spec, d_resources_spec, tick);
        update_resources<<<(RESOURCE_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_resources_spec, tick);
        
        tick_kernel<<<blocks, BLOCK_SIZE>>>(d_agents_uniform, d_resources_uniform, tick);
        update_resources<<<(RESOURCE_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_resources_uniform, tick);
        
        if (tick % 100 == 0) {
            cudaDeviceSynchronize();
            printf("Tick %d/500\r", tick);
            fflush(stdout);
        }
    }
    
    cudaDeviceSynchronize();
    printf("\n\n");
    
   