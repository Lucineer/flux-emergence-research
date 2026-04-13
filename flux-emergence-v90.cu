
/*
CUDA Simulation Experiment v90: STIGMERGY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents (ratio > 1.61x from v8 baseline).
Mechanism: When an agent collects a resource, it deposits pheromone at that location.
           Pheromones decay exponentially (half-life 50 ticks). Agents can detect
           pheromones within their detection range and are attracted to strongest signal.
           Specialists should follow trails more efficiently than uniform agents.
Baseline: Includes v8 confirmed mechanisms (scarcity, territory, communication).
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const float WORLD_SIZE = 1.0f;
const float MIN_DETECT = 0.03f;
const float MAX_DETECT = 0.07f;
const float MIN_GRAB = 0.02f;
const float MAX_GRAB = 0.04f;
const float COMM_RANGE = 0.06f;
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;
const float ANTI_CONVERGE_THRESH = 0.9f;
const float ANTI_CONVERGE_DRIFT = 0.01f;
const float ENERGY_DECAY = 0.999f;
const float PERTURB_PROB = 0.001f;
const float DEFEND_BOOST = 0.2f;
const float SPECIALIST_ROLE = 0.7f;
const float UNIFORM_ROLE = 0.25f;

// Pheromone constants (NOVEL MECHANISM)
const float PHEROMONE_DEPOSIT = 1.0f;
const float PHEROMONE_DECAY = 0.986f; // half-life ~50 ticks
const float PHEROMONE_DETECT_FACTOR = 1.5f; // Increases detection range for pheromones

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPES];
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

// Pheromone struct (NOVEL)
struct Pheromone {
    float x, y;
    float strength;
    int arch; // Which archetype deposited it
};

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents
__global__ void init_agents(Agent* agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    a->rng = idx * 17 + 12345;
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    a->arch = idx % ARCHETYPES;
    
    if (specialized) {
        // Specialists: high value in own archetype role
        for (int i = 0; i < ARCHETYPES; i++) {
            a->role[i] = (i == a->arch) ? SPECIALIST_ROLE : (1.0f - SPECIALIST_ROLE) / (ARCHETYPES - 1);
        }
    } else {
        // Uniform: all roles equal
        for (int i = 0; i < ARCHETYPES; i++) {
            a->role[i] = UNIFORM_ROLE;
        }
    }
}

// Initialize resources
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = idx * 19 + 67890;
    resources[idx].x = lcgf(&rng);
    resources[idx].y = lcgf(&rng);
    resources[idx].value = 0.5f + lcgf(&rng) * 0.5f;
    resources[idx].collected = 0;
}

// Initialize pheromones (NOVEL)
__global__ void init_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES * 4) return; // Max pheromone slots
    
    pheromones[idx].x = 0.0f;
    pheromones[idx].y = 0.0f;
    pheromones[idx].strength = 0.0f;
    pheromones[idx].arch = -1;
}

// Find nearest resource
__device__ int find_nearest_resource(float x, float y, Resource* resources, float max_dist) {
    int nearest = -1;
    float best_dist = max_dist * max_dist;
    
    for (int i = 0; i < RESOURCES; i++) {
        if (resources[i].collected) continue;
        float dx = resources[i].x - x;
        float dy = resources[i].y - y;
        float dist = dx * dx + dy * dy;
        if (dist < best_dist) {
            best_dist = dist;
            nearest = i;
        }
    }
    return nearest;
}

// Find strongest pheromone (NOVEL)
__device__ int find_strongest_pheromone(float x, float y, int arch, Pheromone* pheromones, float max_dist) {
    int strongest = -1;
    float best_strength = 0.0f;
    float max_dist2 = max_dist * max_dist;
    
    for (int i = 0; i < RESOURCES * 4; i++) {
        if (pheromones[i].strength <= 0.0f) continue;
        float dx = pheromones[i].x - x;
        float dy = pheromones[i].y - y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 < max_dist2 && pheromones[i].strength > best_strength) {
            // Prefer pheromones from same archetype
            float strength = pheromones[i].strength * (pheromones[i].arch == arch ? 1.5f : 1.0f);
            if (strength > best_strength) {
                best_strength = strength;
                strongest = i;
            }
        }
    }
    return strongest;
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromones, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= ENERGY_DECAY;
    
    // Random perturbation
    if (lcgf(&a->rng) < PERTURB_PROB) {
        // Defenders resist perturbation
        float resist = 1.0f - a->role[3] * 0.8f;
        a->energy *= (0.5f + 0.5f * resist);
        a->vx += lcgf(&a->rng) * 0.1f - 0.05f;
        a->vy += lcgf(&a->rng) * 0.1f - 0.05f;
    }
    
    // Calculate role-based parameters
    float detect_range = MIN_DETECT + a->role[0] * (MAX_DETECT - MIN_DETECT);
    float grab_range = MIN_GRAB + a->role[1] * (MAX_GRAB - MIN_GRAB);
    
    // Pheromone-enhanced detection (NOVEL)
    float pheromone_detect_range = detect_range * PHEROMONE_DETECT_FACTOR;
    
    // Find nearest resource
    int nearest_res = find_nearest_resource(a->x, a->y, resources, detect_range);
    
    // Find strongest pheromone (NOVEL)
    int strongest_pheromone = find_strongest_pheromone(a->x, a->y, a->arch, pheromones, pheromone_detect_range);
    
    // Movement decision: prioritize pheromones over direct detection
    float target_x = a->x + a->vx;
    float target_y = a->y + a->vy;
    
    if (strongest_pheromone >= 0) {
        // Move toward strongest pheromone
        Pheromone* p = &pheromones[strongest_pheromone];
        target_x = p->x;
        target_y = p->y;
    } else if (nearest_res >= 0) {
        // Move toward nearest resource
        target_x = resources[nearest_res].x;
        target_y = resources[nearest_res].y;
    }
    
    // Move toward target
    float dx = target_x - a->x;
    float dy = target_y - a->y;
    float dist = sqrtf(dx * dx + dy * dy + 1e-6f);
    a->vx += dx / dist * 0.01f;
    a->vy += dy / dist * 0.01f;
    
    // Velocity damping
    a->vx *= 0.95f;
    a->vy *= 0.95f;
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // World boundaries
    if (a->x < 0) { a->x = 0; a->vx = -a->vx; }
    if (a->x > WORLD_SIZE) { a->x = WORLD_SIZE; a->vx = -a->vx; }
    if (a->y < 0) { a->y = 0; a->vy = -a->vy; }
    if (a->y > WORLD_SIZE) { a->y = WORLD_SIZE; a->vy = -a->vy; }
    
    // Resource collection
    if (nearest_res >= 0) {
        Resource* r = &resources[nearest_res];
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < grab_range * grab_range && !r->collected) {
            // Collector bonus
            float bonus = 1.0f + a->role[1] * 0.5f;
            float value = r->value * bonus;
            
            // Defender territory boost
            int defenders_nearby = 0;
            for (int i = 0; i < AGENTS; i++) {
                if (i == idx) continue;
                Agent* other = &agents[i];
                if (other->arch != a->arch) continue;
                float odx = other->x - a->x;
                float ody = other->y - a->y;
                if (odx * odx + ody * ody < 0.04f && other->role[3] > 0.5f) {
                    defenders_nearby++;
                }
            }
            value *= 1.0f + defenders_nearby * DEFEND_BOOST;
            
            a->energy += value;
            a->fitness += value;
            r->collected = 1;
            
            // Deposit pheromone at resource location (NOVEL)
            for (int i = 0; i < RESOURCES * 4; i++) {
                if (pheromones[i].strength <= 0.0f) {
                    pheromones[i].x = r->x;
                    pheromones[i].y = r->y;
                    pheromones[i].strength = PHEROMONE_DEPOSIT;
                    pheromones[i].arch = a->arch;
                    break;
                }
            }
        }
    }
    
    // Communication
    if (a->role[2] > 0.3f && nearest_res >= 0) {
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            if (dx * dx + dy * dy < COMM_RANGE * COMM_RANGE) {
                // Influence neighbor's movement toward resource
                float influence = a->role[2] * 0.1f;
                Resource* r = &resources[nearest_res];
                other->vx += (r->x - other->x) * influence;
                other->vy += (r->y - other->y) * influence;
            }
        }
    }
    
    // Social coupling
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent* other = &agents[i];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < 0.04f) {
            float coupling = (a->arch == other->arch) ? COUPLING_SAME : COUPLING_DIFF;
            
            for (int r = 0; r < ARCHETYPES; r++) {
                float diff = other->role[r] - a->role[r];
                a->role[r] += diff * coupling;
                other->role[r] -= diff * coupling;
            }
        }
    }
    
    // Anti-convergence
    float max_role = 0.0f;
    int max_idx = 0;
    for (int r = 0; r < ARCHETYPES; r++) {
        if (a->role[r] > max_role) {
            max_role = a->role[r];
            max_idx = r;
        }
    }
    
    if (max_role > ANTI_CONVERGE_THRESH) {
        // Apply random drift to non-dominant roles
        for (int r = 0; r < ARCHETYPES; r++) {
            if (r != max_idx) {
                float drift = (lcgf(&a->rng) * 2.0f - 1.0f) * ANTI_CONVERGE_DRIFT;
                a->role[r] += drift;
                if (a->role[r] < 0) a->role[r] = 0;
            }
        }
        
        // Renormalize
        float sum = 0.0f;
        for (int r = 0; r < ARCHETYPES; r++) sum += a->role[r];
        for (int r = 0; r < ARCHETYPES; r++) a->role[r] /= sum;
    }
}

// Decay pheromones (NOVEL)
__global__ void decay_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES * 4) return;
    
    pheromones[idx].strength *= PHEROMONE_DECAY;
    if (pheromones[idx].strength < 0.001f) {
        pheromones[idx].strength = 0.0f;
        pheromones[idx].arch = -1;
    }
}

// Respawn resources
__global__ void respawn_resources(Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    // Respawn every 50 ticks
    if (tick_num % 50 == 0) {
        unsigned int rng = idx * 19 + tick_num * 7919;
        resources[idx].x = lcgf(&rng);
        resources[idx].y = lcgf(&rng);
        resources[idx].value = 0.5f + lcgf(&rng) * 0.5f;
        resources[idx].collected = 0;
    }
}

int main() {
    printf("=== CUDA Experiment v90: STIGMERGY TRAILS ===\n");
    printf("Testing: Pheromone trails at resource locations\n");
    printf("Prediction: Specialists >1.61x advantage over uniform (v8 baseline=1.61x)\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RESOURCES, TICKS);
    
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    Pheromone* d_pheromones_spec;
    Pheromone* d_pheromones_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_phe