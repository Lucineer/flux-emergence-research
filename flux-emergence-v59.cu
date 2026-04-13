// CUDA Simulation Experiment v59: STIGMERGY TRAILS
// Testing: Agents leave pheromone trails at resource locations that decay over time
// Prediction: Pheromones will improve specialist efficiency by 20-30% over baseline v8
// Mechanism: When agent collects resource, leaves pheromone at location (strength=1.0)
// Pheromones decay by 0.01 per tick, visible to same-arch agents within 0.08 range
// Pheromones attract agents (add small velocity component toward strongest nearby trail)
// Control group: No pheromone trails (same as v8 baseline)
// Expected: Specialists better at following trails to recent resource locations

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RES_COUNT = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Agent archetypes
enum { ARCH_GENERALIST = 0, ARCH_SPECIALIST = 1 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role strengths: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Archetype: 0=generalist, 1=specialist
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone structure (NEW for v59)
struct Pheromone {
    float x, y;           // Location
    float strength;       // Current strength
    int arch;             // Archetype that left it
    int age;              // Age in ticks
};

// Linear Congruential Generator
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return (lcg(state) & 0xFFFF) / 65535.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, int arch, float* role_template) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    unsigned int seed = idx * 137 + 7919;
    agents[idx].x = lcgf(&seed) * 2.0f - 1.0f;
    agents[idx].y = lcgf(&seed) * 2.0f - 1.0f;
    agents[idx].vx = (lcgf(&seed) * 2.0f - 1.0f) * 0.01f;
    agents[idx].vy = (lcgf(&seed) * 2.0f - 1.0f) * 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].arch = arch;
    agents[idx].rng = idx * 7919 + 137;
    
    // Set roles based on archetype
    if (arch == ARCH_SPECIALIST) {
        // Specialists have strong primary role based on idx % 4
        int primary = idx % 4;
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = (i == primary) ? 0.7f : 0.1f;
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
    if (idx >= RES_COUNT) return;
    
    unsigned int seed = idx * 7919 + 137;
    resources[idx].x = lcgf(&seed) * 2.0f - 1.0f;
    resources[idx].y = lcgf(&seed) * 2.0f - 1.0f;
    resources[idx].value = 0.5f + lcgf(&seed) * 0.5f;  // 0.5-1.0
    resources[idx].collected = 0;
}

// Initialize pheromones kernel (NEW for v59)
__global__ void init_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT * 4) return;  // Space for up to 4x resources
    
    pheromones[idx].strength = 0.0f;
    pheromones[idx].age = 1000;  // Old = inactive
}

// Find nearest resource
__device__ int find_nearest_resource(float x, float y, Resource* resources, float max_dist) {
    int nearest = -1;
    float min_dist = max_dist * max_dist;
    
    for (int i = 0; i < RES_COUNT; i++) {
        if (resources[i].collected) continue;
        float dx = resources[i].x - x;
        float dy = resources[i].y - y;
        float dist = dx * dx + dy * dy;
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    return nearest;
}

// Find strongest pheromone (NEW for v59)
__device__ int find_strongest_pheromone(float x, float y, int arch, Pheromone* pheromones, 
                                        float max_dist, float* out_strength) {
    int strongest = -1;
    float max_strength = 0.0f;
    
    for (int i = 0; i < RES_COUNT * 4; i++) {
        if (pheromones[i].strength < 0.1f) continue;  // Too weak
        if (pheromones[i].arch != arch) continue;     // Only same archetype
        
        float dx = pheromones[i].x - x;
        float dy = pheromones[i].y - y;
        float dist = dx * dx + dy * dy;
        if (dist < max_dist * max_dist && pheromones[i].strength > max_strength) {
            max_strength = pheromones[i].strength;
            strongest = i;
        }
    }
    
    *out_strength = max_strength;
    return strongest;
}

// Add pheromone at location (NEW for v59)
__device__ void add_pheromone(float x, float y, int arch, Pheromone* pheromones) {
    // Find oldest pheromone slot to replace
    int oldest_idx = 0;
    int max_age = -1;
    
    for (int i = 0; i < RES_COUNT * 4; i++) {
        if (pheromones[i].age > max_age) {
            max_age = pheromones[i].age;
            oldest_idx = i;
        }
    }
    
    pheromones[oldest_idx].x = x;
    pheromones[oldest_idx].y = y;
    pheromones[oldest_idx].strength = 1.0f;  // Full strength
    pheromones[oldest_idx].arch = arch;
    pheromones[oldest_idx].age = 0;
}

// Update pheromones (decay and age) (NEW for v59)
__global__ void update_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT * 4) return;
    
    if (pheromones[idx].strength > 0.0f) {
        pheromones[idx].strength *= 0.99f;  // Decay 1% per tick
        pheromones[idx].age++;
        
        // Remove if too weak
        if (pheromones[idx].strength < 0.01f) {
            pheromones[idx].strength = 0.0f;
            pheromones[idx].age = 1000;
        }
    }
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromones, 
                     int use_pheromones, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other = (idx + 37) % AGENT_COUNT;
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a.role[i] - agents[other].role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant role
        int dominant = 0;
        for (int i = 1; i < 4; i++) {
            if (a.role[i] > a.role[dominant]) dominant = i;
        }
        int drift_role;
        do {
            drift_role = (int)(lcgf(&a.rng) * 4);
        } while (drift_role == dominant);
        
        a.role[drift_role] += (lcgf(&a.rng) * 2.0f - 1.0f) * 0.01f;
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int i = 0; i < 4; i++) a.role[i] /= sum;
    }
    
    // Role-based behavior
    float explore_range = 0.03f + a.role[0] * 0.04f;  // 0.03-0.07
    float collect_range = 0.02f + a.role[1] * 0.02f;  // 0.02-0.04
    float comm_range = 0.04f + a.role[2] * 0.02f;     // 0.04-0.06
    float defend_range = 0.03f + a.role[3] * 0.02f;   // 0.03-0.05
    
    // Find nearest resource
    int nearest_res = find_nearest_resource(a.x, a.y, resources, explore_range);
    
    // Pheromone attraction (NEW for v59)
    float pheromone_strength = 0.0f;
    int nearest_pheromone = -1;
    if (use_pheromones) {
        nearest_pheromone = find_strongest_pheromone(a.x, a.y, a.arch, pheromones, 
                                                    0.08f, &pheromone_strength);
    }
    
    // Movement with pheromone influence
    float target_x = 0.0f, target_y = 0.0f;
    int has_target = 0;
    
    if (nearest_res >= 0) {
        target_x = resources[nearest_res].x;
        target_y = resources[nearest_res].y;
        has_target = 1;
    } else if (nearest_pheromone >= 0 && pheromone_strength > 0.3f) {
        // Follow pheromone trail if strong enough
        target_x = pheromones[nearest_pheromone].x;
        target_y = pheromones[nearest_pheromone].y;
        has_target = 1;
    }
    
    if (has_target) {
        float dx = target_x - a.x;
        float dy = target_y - a.y;
        float dist = sqrtf(dx * dx + dy * dy + 1e-6f);
        dx /= dist;
        dy /= dist;
        
        // Blend with pheromone strength
        float blend = (nearest_pheromone >= 0) ? pheromone_strength : 0.0f;
        a.vx = a.vx * 0.7f + dx * 0.03f * (1.0f + blend * 0.5f);
        a.vy = a.vy * 0.7f + dy * 0.03f * (1.0f + blend * 0.5f);
    } else {
        // Random walk
        a.vx += (lcgf(&a.rng) * 2.0f - 1.0f) * 0.001f;
        a.vy += (lcgf(&a.rng) * 2.0f - 1.0f) * 0.001f;
    }
    
    // Velocity limits
    float speed = sqrtf(a.vx * a.vx + a.vy * a.vy);
    if (speed > 0.02f) {
        a.vx *= 0.02f / speed;
        a.vy *= 0.02f / speed;
    }
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // Boundary check
    if (a.x < -1.0f) { a.x = -1.0f; a.vx = fabsf(a.vx) * 0.5f; }
    if (a.x > 1.0f) { a.x = 1.0f; a.vx = -fabsf(a.vx) * 0.5f; }
    if (a.y < -1.0f) { a.y = -1.0f; a.vy = fabsf(a.vy) * 0.5f; }
    if (a.y > 1.0f) { a.y = 1.0f; a.vy = -fabsf(a.vy) * 0.5f; }
    
    // Resource collection
    if (nearest_res >= 0) {
        Resource& r = resources[nearest_res];
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < collect_range * collect_range) {
            // Collect resource
            float bonus = 1.0f + a.role[1] * 0.5f;  // Up to 50% bonus for collectors
            a.energy += r.value * bonus;
            a.fitness += r.value * bonus;
            r.collected = 1;
            
            // Leave pheromone at resource location (NEW for v59)
            if (use_pheromones) {
                add_pheromone(r.x, r.y, a.arch, pheromones);
            }
        }
    }
    
    // Communication (broadcast nearest resource)
    if (a.role[2] > 0.3f && nearest_res >= 0) {
        // In real implementation, would communicate to nearby agents
        // Simplified for performance
    }
    
    // Defense territory bonus
    int defender_count = 0;
    for (int i = 0; i < AGENT_COUNT; i++) {
        if (i == idx) continue;
        if (agents[i].arch != a.arch) continue;
        float dx = agents[i].x - a.x;
        float dy = agents[i].y - a.y;
        if (dx * dx + dy * dy < defend_range * defend_range) {
            if (agents[i].role[3] > 0.3f) defender_count++;
        }
    }
    
    if (a.role[3] > 0.3f && defender_count > 0) {
        a.energy *= 1.0f + defender_count * 0.2f;  // 20% per nearby defender
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 25) {
        if (a.role[3] < 0.3f) {  // Defenders resist perturbation
            a.energy *= 0.5f;
        }
    }
}

// Reset resources kernel
__global__ void reset_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    if (resources[idx].collected) {
        unsigned int seed = idx * 7919 + 137 + (resources[idx].collected * 123);
        resources[idx].x = lcgf(&seed) * 2.0f - 1.0f;
        resources[idx].y = lcgf(&seed) * 2.0f - 1.0f;
        resources[idx].value = 0.5f + lcgf(&seed) * 0.5f;
        resources[idx].collected = 0;
    }
}

int main() {
    printf("=== CUDA Simulation Experiment v59: STIGMERGY TRAILS ===\n");
    printf("Testing: Pheromone trails at resource locations\n");
    printf("Prediction: 20-30% efficiency improvement for specialists\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENT_COUNT, RES_COUNT, TICKS);
    
