// CUDA Simulation Experiment v61: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone markers at collected resource locations
// Prediction: Pheromones will enhance specialist coordination, increasing advantage ratio >1.61x
// Novelty: Stigmergic communication via persistent environmental markers
// Baseline: v8 mechanisms (scarcity, territory, comms) included

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 256; // Spatial grid for pheromone tracking
const int MAX_PHEROMONES = 2048; // Maximum active pheromone markers

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Dominant archetype (0-3)
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone structure - NEW for v61
struct Pheromone {
    float x, y;           // Location
    float strength;       // Current strength
    int arch;             // Archetype that left it
    int tick_created;     // When it was created
};

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, Resource* resources, Pheromone* pheromones, 
                           int* pheromone_count, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    a->rng = seed + idx * 17;
    
    // Random position
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    
    // Specialized agents (first half) vs uniform control (second half)
    if (idx < AGENTS / 2) {
        // Specialized: one dominant role at 0.7, others at 0.1
        a->arch = idx % 4;  // Distribute archetypes evenly
        for (int i = 0; i < 4; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles at 0.25
        a->arch = 4;  // Uniform archetype
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    unsigned int rng = seed + idx * 13;
    
    // Uniform random distribution (v12 showed powerlaw slightly hurts specialists)
    r->x = lcgf(&rng);
    r->y = lcgf(&rng);
    r->value = 0.5f + lcgf(&rng) * 0.5f;  // Value between 0.5 and 1.0
    r->collected = 0;
}

// Calculate role similarity between agents
__device__ float role_similarity(const Agent* a, const Agent* b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < 4; i++) {
        dot += a->role[i] * b->role[i];
        norm_a += a->role[i] * a->role[i];
        norm_b += b->role[i] * b->role[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Main simulation kernel
__global__ void tick_kernel(Agent* agents, Resource* resources, Pheromone* pheromones,
                           int* pheromone_count, int tick, int* resource_collected) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence mechanism (v3, v13)
    int nearest = -1;
    float max_sim = 0.0f;
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        float sim = role_similarity(a, &agents[i]);
        if (sim > max_sim) {
            max_sim = sim;
            nearest = i;
        }
    }
    
    if (max_sim > 0.9f && nearest != -1) {
        // Find non-dominant role
        int dominant = 0;
        for (int i = 1; i < 4; i++) {
            if (a->role[i] > a->role[dominant]) dominant = i;
        }
        
        int drift_role;
        do {
            drift_role = int(lcgf(&a->rng) * 4);
        } while (drift_role == dominant);
        
        // Apply random drift (v13 robust range)
        float drift = lcgf(&a->rng) * 0.02f - 0.01f;
        a->role[drift_role] += drift;
        a->role[drift_role] = fmaxf(0.05f, fminf(0.8f, a->role[drift_role]));
        
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) a->role[i] /= sum;
    }
    
    // Movement with pheromone influence - NEW for v61
    float pheromone_force_x = 0.0f, pheromone_force_y = 0.0f;
    int pheromone_nearby = 0;
    
    // Check nearby pheromones (stigmergy)
    for (int i = 0; i < *pheromone_count; i++) {
        Pheromone* p = &pheromones[i];
        if (p->strength <= 0.0f) continue;
        
        float dx = p->x - a->x;
        float dy = p->y - a->y;
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < 0.04f) {  // Detection range for pheromones
            // Stronger attraction to pheromones from same archetype
            float attraction = (a->arch == p->arch || a->arch == 4) ? 1.0f : 0.3f;
            float force = p->strength * attraction / (dist2 + 0.01f);
            
            pheromone_force_x += dx * force;
            pheromone_force_y += dy * force;
            pheromone_nearby++;
        }
    }
    
    // Normalize pheromone influence
    if (pheromone_nearby > 0) {
        pheromone_force_x /= pheromone_nearby;
        pheromone_force_y /= pheromone_nearby;
        float mag = sqrtf(pheromone_force_x * pheromone_force_x + 
                         pheromone_force_y * pheromone_force_y);
        if (mag > 0.01f) {
            pheromone_force_x = pheromone_force_x / mag * 0.01f;
            pheromone_force_y = pheromone_force_y / mag * 0.01f;
        }
    }
    
    // Update velocity with pheromone influence and random walk
    a->vx = a->vx * 0.8f + (lcgf(&a->rng) * 0.02f - 0.01f) + pheromone_force_x;
    a->vy = a->vy * 0.8f + (lcgf(&a->rng) * 0.02f - 0.01f) + pheromone_force_y;
    
    // Limit speed
    float speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
    if (speed > 0.02f) {
        a->vx *= 0.02f / speed;
        a->vy *= 0.02f / speed;
    }
    
    // Update position with wrap-around
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0.0f) a->x += 1.0f;
    if (a->x > 1.0f) a->x -= 1.0f;
    if (a->y < 0.0f) a->y += 1.0f;
    if (a->y > 1.0f) a->y -= 1.0f;
    
    // Resource interaction
    float best_value = 0.0f;
    int best_resource = -1;
    float best_dx = 0.0f, best_dy = 0.0f;
    
    // Explore role: detection range 0.03-0.07
    float detect_range = 0.03f + a->role[0] * 0.04f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < detect_range * detect_range) {
            float value = r->value;
            
            // Collect role: grab range 0.02-0.04 with 50% bonus
            float grab_range = 0.02f + a->role[1] * 0.02f;
            if (dist2 < grab_range * grab_range) {
                value *= 1.5f;  // Collection bonus
            }
            
            if (value > best_value) {
                best_value = value;
                best_resource = i;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    
    // Collect resource if in grab range
    if (best_resource != -1) {
        Resource* r = &resources[best_resource];
        float grab_range = 0.02f + a->role[1] * 0.02f;
        float dist2 = best_dx * best_dx + best_dy * best_dy;
        
        if (dist2 < grab_range * grab_range && !r->collected) {
            // Collect resource
            float collected_value = r->value;
            
            // Territory bonus from nearby defenders (v8)
            int defenders_nearby = 0;
            for (int i = 0; i < AGENTS; i++) {
                if (i == idx) continue;
                Agent* other = &agents[i];
                if (other->arch != ARCH_DEFENDER && a->arch != ARCH_DEFENDER) continue;
                
                float odx = other->x - a->x;
                float ody = other->y - a->y;
                if (odx > 0.5f) odx -= 1.0f;
                if (odx < -0.5f) odx += 1.0f;
                if (ody > 0.5f) ody -= 1.0f;
                if (ody < -0.5f) ody += 1.0f;
                
                if (odx * odx + ody * ody < 0.04f) {
                    defenders_nearby++;
                }
            }
            
            // 20% bonus per nearby defender
            collected_value *= (1.0f + defenders_nearby * 0.2f);
            
            a->energy += collected_value;
            a->fitness += collected_value;
            r->collected = 1;
            atomicAdd(resource_collected, 1);
            
            // Leave pheromone at collected location - NEW for v61
            int p_idx = atomicAdd(pheromone_count, 1);
            if (p_idx < MAX_PHEROMONES) {
                Pheromone* p = &pheromones[p_idx];
                p->x = r->x;
                p->y = r->y;
                p->strength = 1.0f;  // Full strength initially
                p->arch = a->arch;
                p->tick_created = tick;
            }
        }
    }
    
    // Communicate role: broadcast nearest resource location
    if (a->role[2] > 0.3f && best_resource != -1) {
        float comm_range = 0.06f;
        Resource* r = &resources[best_resource];
        
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            if (dx > 0.5f) dx -= 1.0f;
            if (dx < -0.5f) dx += 1.0f;
            if (dy > 0.5f) dy -= 1.0f;
            if (dy < -0.5f) dy += 1.0f;
            
            if (dx * dx + dy * dy < comm_range * comm_range) {
                // Influence neighbor's movement toward resource
                float influence = a->role[2] * 0.01f;
                other->vx += (r->x - other->x) * influence;
                other->vy += (r->y - other->y) * influence;
            }
        }
    }
    
    // Defend role: perturbation resistance
    if (a->role[3] > 0.3f) {
        // Defenders resist energy loss from perturbations
        if (tick % 100 == idx % 100) {  // Periodic perturbations
            float resistance = a->role[3];
            a->energy *= (0.5f + resistance * 0.5f);  // 50-100% resistance
        }
    }
    
    // Coupling: role adjustment toward similar agents
    float coupling_strength = 0.0f;
    int similar_count = 0;
    
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        
        Agent* other = &agents[i];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist2 = dx * dx + dy * dy;
        if (dist2 < 0.04f) {  // Interaction range
            float sim = role_similarity(a, other);
            if (sim > 0.7f) {
                coupling_strength += (a->arch == other->arch) ? 0.02f : 0.002f;
                similar_count++;
                
                // Adjust roles toward neighbor
                for (int j = 0; j < 4; j++) {
                    float diff = other->role[j] - a->role[j];
                    a->role[j] += diff * 0.001f;
                }
            }
        }
    }
    
    // Renormalize roles
    if (similar_count > 0) {
        float sum = a->role[0] + a->role[1] + a->