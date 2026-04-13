// CUDA Simulation Experiment v18: STIGMERGY TRAILS
// Testing: Pheromone trails at resource locations that decay over time
// Prediction: Stigmergy will improve specialist efficiency by 20-30% (1.9-2.1x ratio)
// Mechanism: Agents deposit pheromone when collecting resources, others follow trails
// Baseline: v8 confirmed mechanisms (scarcity, territory, comms) + anti-convergence

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK = 256;

// Agent archetypes
enum { ARCH_GENERALIST = 0, ARCH_SPECIALIST = 1 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: [explore, collect, communicate, defend]
    float fitness;        // Fitness score
    int arch;             // Archetype
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
    unsigned int rng;     // RNG state for respawning
};

// Pheromone structure for stigmergy
struct Pheromone {
    float x, y;           // Location
    float strength;       // Strength (0-1)
    int arch;             // Archetype that deposited it
    int age;              // Age in ticks
};

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int arch, float* role_template) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    unsigned int seed = idx * 17 + arch * 7919;
    agents[idx].x = lcgf(&seed);
    agents[idx].y = lcgf(&seed);
    agents[idx].vx = lcgf(&seed) * 0.02 - 0.01;
    agents[idx].vy = lcgf(&seed) * 0.02 - 0.01;
    agents[idx].energy = 1.0f;
    agents[idx].arch = arch;
    agents[idx].rng = seed;
    agents[idx].fitness = 0.0f;
    
    // Set roles based on archetype
    if (arch == ARCH_SPECIALIST) {
        // Specialists have dominant role based on thread index
        int dominant = idx % 4;
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = (i == dominant) ? 0.7f : 0.1f;
        }
    } else {
        // Generalists have uniform roles
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int seed = idx * 19 + 137;
    resources[idx].x = lcgf(&seed);
    resources[idx].y = lcgf(&seed);
    resources[idx].value = 0.5f + lcgf(&seed) * 0.5f;
    resources[idx].collected = 0;
    resources[idx].rng = seed;
}

// Initialize pheromones kernel
__global__ void init_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES * 4) return;  // Space for pheromones at resources
    
    pheromones[idx].x = 0;
    pheromones[idx].y = 0;
    pheromones[idx].strength = 0;
    pheromones[idx].arch = -1;
    pheromones[idx].age = 0;
}

// Find nearest resource
__device__ int find_nearest_resource(float x, float y, Resource* resources, 
                                    float max_dist, float* dist_out) {
    int nearest = -1;
    float min_dist = max_dist;
    
    for (int i = 0; i < RESOURCES; i++) {
        if (resources[i].collected) continue;
        
        float dx = x - resources[i].x;
        float dy = y - resources[i].y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    
    *dist_out = min_dist;
    return nearest;
}

// Find strongest pheromone in range
__device__ int find_strongest_pheromone(float x, float y, Pheromone* pheromones, 
                                       int pheromone_count, int same_arch) {
    int strongest = -1;
    float max_strength = 0.1f;  // Minimum threshold
    
    for (int i = 0; i < pheromone_count; i++) {
        if (pheromones[i].strength < 0.01f) continue;
        if (same_arch && pheromones[i].arch != same_arch) continue;
        
        float dx = x - pheromones[i].x;
        float dy = y - pheromones[i].y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.15f) {  // Pheromone detection range
            float adjusted = pheromones[i].strength / (dist + 0.01f);
            if (adjusted > max_strength) {
                max_strength = adjusted;
                strongest = i;
            }
        }
    }
    
    return strongest;
}

// Deposit pheromone at location
__device__ void deposit_pheromone(float x, float y, Pheromone* pheromones, 
                                 int pheromone_count, int arch) {
    // Find oldest or weakest pheromone to replace
    int replace_idx = -1;
    float min_strength = 2.0f;
    int max_age = -1;
    
    for (int i = 0; i < pheromone_count; i++) {
        if (pheromones[i].strength < 0.01f) {
            replace_idx = i;
            break;
        }
        if (pheromones[i].strength < min_strength) {
            min_strength = pheromones[i].strength;
            replace_idx = i;
        }
        if (pheromones[i].age > max_age) {
            max_age = pheromones[i].age;
        }
    }
    
    if (replace_idx >= 0) {
        pheromones[replace_idx].x = x;
        pheromones[replace_idx].y = y;
        pheromones[replace_idx].strength = 1.0f;  // Full strength
        pheromones[replace_idx].arch = arch;
        pheromones[replace_idx].age = 0;
    }
}

// Update pheromones (decay and age)
__global__ void update_pheromones(Pheromone* pheromones, int pheromone_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pheromone_count) return;
    
    if (pheromones[idx].strength > 0) {
        // Decay strength
        pheromones[idx].strength *= 0.97f;  // 3% decay per tick
        pheromones[idx].age++;
        
        // Remove very old or weak pheromones
        if (pheromones[idx].strength < 0.01f || pheromones[idx].age > 100) {
            pheromones[idx].strength = 0;
            pheromones[idx].arch = -1;
        }
    }
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromones, 
                    int tick_num, int* perturbations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Random perturbation (10% chance every 50 ticks)
    if (tick_num % 50 == 0 && lcgf(&a->rng) < 0.1f) {
        // Defenders resist perturbation
        float resist = 1.0f - a->role[3] * 0.8f;
        if (lcgf(&a->rng) > resist) {
            a->energy *= 0.5f;
            atomicAdd(perturbations, 1);
        }
    }
    
    // Anti-convergence: prevent role homogenization
    float max_role = 0;
    int max_idx = 0;
    for (int i = 0; i < 4; i++) {
        if (a->role[i] > max_role) {
            max_role = a->role[i];
            max_idx = i;
        }
    }
    
    if (max_role > 0.9f) {
        // Apply random drift to non-dominant roles
        int drift_idx = (max_idx + 1 + (int)(lcgf(&a->rng) * 3)) % 4;
        float drift = lcgf(&a->rng) * 0.02f - 0.01f;
        a->role[drift_idx] = fmaxf(0.05f, fminf(0.95f, a->role[drift_idx] + drift));
        
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) {
            a->role[i] /= sum;
        }
    }
    
    // Find nearby agents for coupling and communication
    int nearby_same = 0;
    int nearby_diff = 0;
    float comm_x = 0, comm_y = 0;
    int comm_count = 0;
    
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        
        float dx = a->x - agents[i].x;
        float dy = a->y - agents[i].y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.06f) {  // Communication range
            if (agents[i].arch == a->arch) {
                nearby_same++;
                
                // Communicate resource locations
                float res_dist;
                int res_idx = find_nearest_resource(agents[i].x, agents[i].y, 
                                                   resources, 0.2f, &res_dist);
                if (res_idx >= 0 && res_dist < 0.1f) {
                    comm_x += resources[res_idx].x;
                    comm_y += resources[res_idx].y;
                    comm_count++;
                }
            } else {
                nearby_diff++;
            }
        }
        
        // Role coupling
        if (dist < 0.02f) {
            float coupling = (agents[i].arch == a->arch) ? 0.02f : 0.002f;
            for (int r = 0; r < 4; r++) {
                float diff = agents[i].role[r] - a->role[r];
                a->role[r] += diff * coupling;
            }
        }
    }
    
    // Normalize roles after coupling
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) {
        a->role[i] /= sum;
    }
    
    // STIGMERGY: Follow pheromone trails (novel mechanism)
    float pheromone_influence = 0;
    float target_x = 0, target_y = 0;
    
    int strong_pheromone = find_strongest_pheromone(a->x, a->y, pheromones, 
                                                   RESOURCES * 4, a->arch);
    
    if (strong_pheromone >= 0) {
        pheromone_influence = pheromones[strong_pheromone].strength * a->role[0] * 0.5f;
        target_x = pheromones[strong_pheromone].x;
        target_y = pheromones[strong_pheromone].y;
    }
    
    // Use communicated information
    if (comm_count > 0 && a->role[2] > 0.2f) {
        target_x = comm_x / comm_count;
        target_y = comm_y / comm_count;
        pheromone_influence = 0;  // Communication overrides pheromones
    }
    
    // Movement decision
    float move_x = 0, move_y = 0;
    
    if (pheromone_influence > 0 || comm_count > 0) {
        // Move toward target (pheromone or communicated resource)
        float dx = target_x - a->x;
        float dy = target_y - a->y;
        float dist = sqrtf(dx*dx + dy*dy) + 0.001f;
        move_x += dx / dist * 0.03f;
        move_y += dy / dist * 0.03f;
    } else {
        // Explore randomly
        move_x += (lcgf(&a->rng) - 0.5f) * 0.02f * a->role[0];
        move_y += (lcgf(&a->rng) - 0.5f) * 0.02f * a->role[0];
    }
    
    // Update position with velocity damping
    a->vx = a->vx * 0.9f + move_x;
    a->vy = a->vy * 0.9f + move_y;
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary check
    if (a->x < 0) { a->x = 0; a->vx *= -0.5f; }
    if (a->x > 1) { a->x = 1; a->vx *= -0.5f; }
    if (a->y < 0) { a->y = 0; a->vy *= -0.5f; }
    if (a->y > 1) { a->y = 1; a->vy *= -0.5f; }
    
    // Resource collection
    float collect_range = 0.02f + a->role[1] * 0.02f;
    float detect_range = 0.03f + a->role[0] * 0.04f;
    
    float nearest_dist;
    int nearest = find_nearest_resource(a->x, a->y, resources, detect_range, &nearest_dist);
    
    if (nearest >= 0 && nearest_dist < collect_range) {
        // Collect resource
        float base_value = resources[nearest].value;
        
        // Collection bonus for specialists
        float bonus = 1.0f + a->role[1] * 0.5f;
        
        // Territory bonus from nearby defenders of same archetype
        float territory_bonus = 1.0f + nearby_same * a->role[3] * 0.2f;
        
        float value = base_value * bonus * territory_bonus;
        a->energy += value;
        a->fitness += value;
        
        // STIGMERGY: Deposit pheromone at collected resource location
        deposit_pheromone(resources[nearest].x, resources[nearest].y, 
                         pheromones, RESOURCES * 4, a->arch);
        
        // Mark resource as collected
        resources[nearest].collected = 1;
    }
    
    // Energy limits
    if (a->energy > 2.0f) a->energy = 2.0f;
    if (a->energy < 0) a->energy = 0;
}

// Reset collected resources
__global__ void reset_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    // 10% chance to respawn each tick
    if (resources[idx].collected && lcgf(&resources[idx].rng) < 0.1f) {
        resources[idx].collected = 0;
        resources[idx].x = lcgf(&resources[idx].rng);
        resources[idx].y = lcgf(&resources[idx].rng);
    }
}

int main() {
    printf("=== CUDA Simulation Experiment v18: STIGMERGY TRAILS ===\n");
    printf("Testing: Pheromone trails at resource locations\n");
    printf("Prediction: Stigmergy improves specialist efficiency by 20-30%% (1.9-2.1x ratio)\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RES