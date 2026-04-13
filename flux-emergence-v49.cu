/*
CUDA Simulation Experiment v49: STIGMERY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents from 1.61x to >2.0x due to amplified information sharing.
Baseline: v8 mechanisms (scarcity, territory, communication) included.
Novelty: Stigmergy via pheromone trails with exponential decay.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants for sm_87 (Jetson Orin)
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Environment constants
const float WORLD_SIZE = 1.0f;
const float MIN_COORD = 0.0f;
const float MAX_COORD = 1.0f;

// Role indices
const int ROLE_EXPLORE = 0;
const int ROLE_COLLECT = 1;
const int ROLE_COMM = 2;
const int ROLE_DEFEND = 3;

// Archetype constants
const int ARCH_SPECIALIST = 0;
const int ARCH_UNIFORM = 1;
const int ARCH_COUNT = 2;

// Baseline v8 parameters
const float DETECT_RANGE_MIN = 0.03f;
const float DETECT_RANGE_MAX = 0.07f;
const float GRAB_RANGE_MIN = 0.02f;
const float GRAB_RANGE_MAX = 0.04f;
const float COMM_RANGE = 0.06f;
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;
const float ENERGY_DECAY = 0.999f;
const float PERTURB_ENERGY_LOSS = 0.5f;
const float DEFEND_BOOST_PER_NEIGHBOR = 0.2f;
const float ANTI_CONVERGENCE_THRESH = 0.9f;
const float ANTI_CONVERGENCE_DRIFT = 0.01f;
const float TERRITORY_RANGE = 0.08f;

// v49 Novel: Stigmergy parameters
const float PHEROMONE_DROP_STRENGTH = 0.3f;  // Strength when dropping pheromone
const float PHEROMONE_DECAY_RATE = 0.95f;    // Per-tick decay
const float PHEROMONE_SENSE_RANGE = 0.1f;    // How far agents sense pheromones
const float PHEROMONE_INFLUENCE = 0.5f;      // How much pheromones influence movement

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles
    float fitness;        // Fitness score
    int arch;             // Archetype (0=specialist, 1=uniform)
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Whether collected
};

// Pheromone structure (v49 novel)
struct Pheromone {
    float x, y;           // Location
    float strength;       // Current strength
    int arch;             // Which archetype dropped it
};

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent *agents, int arch, float *role_template) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS/ARCH_COUNT) return;
    
    int agent_idx = idx + (arch * (AGENTS/ARCH_COUNT));
    unsigned int seed = agent_idx * 123456789 + 987654321;
    
    agents[agent_idx].x = lcgf(seed) * WORLD_SIZE;
    agents[agent_idx].y = lcgf(seed) * WORLD_SIZE;
    agents[agent_idx].vx = (lcgf(seed) - 0.5f) * 0.01f;
    agents[agent_idx].vy = (lcgf(seed) - 0.5f) * 0.01f;
    agents[agent_idx].energy = 1.0f;
    agents[agent_idx].fitness = 0.0f;
    agents[agent_idx].arch = arch;
    agents[agent_idx].rng = seed;
    
    // Set roles based on archetype
    for (int i = 0; i < 4; i++) {
        if (arch == ARCH_SPECIALIST) {
            // Specialists: one dominant role (0.7), others 0.1 each
            agents[agent_idx].role[i] = (i == (idx % 4)) ? 0.7f : 0.1f;
        } else {
            // Uniform: all roles equal
            agents[agent_idx].role[i] = 0.25f;
        }
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int seed = idx * 135791113 + 17192123;
    resources[idx].x = lcgf(seed) * WORLD_SIZE;
    resources[idx].y = lcgf(seed) * WORLD_SIZE;
    resources[idx].value = 0.5f + lcgf(seed) * 0.5f;  // 0.5 to 1.0
    resources[idx].collected = 0;
}

// Initialize pheromones kernel (v49 novel)
__global__ void init_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES * 2) return;  // Each resource can have 2 pheromones (one per arch)
    
    pheromones[idx].x = 0.0f;
    pheromones[idx].y = 0.0f;
    pheromones[idx].strength = 0.0f;
    pheromones[idx].arch = idx % ARCH_COUNT;
}

// Find nearest resource
__device__ int find_nearest_resource(float x, float y, Resource *resources, 
                                     float max_range, float &dist_sq) {
    int nearest = -1;
    float min_dist_sq = max_range * max_range;
    
    for (int i = 0; i < RESOURCES; i++) {
        if (resources[i].collected) continue;
        
        float dx = resources[i].x - x;
        float dy = resources[i].y - y;
        float d2 = dx*dx + dy*dy;
        
        if (d2 < min_dist_sq) {
            min_dist_sq = d2;
            nearest = i;
        }
    }
    
    dist_sq = min_dist_sq;
    return nearest;
}

// Find strongest pheromone in range (v49 novel)
__device__ int find_strongest_pheromone(float x, float y, Pheromone *pheromones, 
                                        int arch, float &attract_x, float &attract_y) {
    int strongest = -1;
    float max_strength = 0.0f;
    float range_sq = PHEROMONE_SENSE_RANGE * PHEROMONE_SENSE_RANGE;
    
    for (int i = 0; i < RESOURCES * 2; i++) {
        if (pheromones[i].strength < 0.01f) continue;
        if (pheromones[i].arch != arch) continue;  // Only follow own archetype's trails
        
        float dx = pheromones[i].x - x;
        float dy = pheromones[i].y - y;
        float d2 = dx*dx + dy*dy;
        
        if (d2 < range_sq && pheromones[i].strength > max_strength) {
            max_strength = pheromones[i].strength;
            strongest = i;
            attract_x = pheromones[i].x;
            attract_y = pheromones[i].y;
        }
    }
    
    return strongest;
}

// Main simulation kernel
__global__ void tick_kernel(Agent *agents, Resource *resources, Pheromone *pheromones, 
                           int tick, float *arch_energy_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Apply energy decay
    a.energy *= ENERGY_DECAY;
    
    // Apply anti-convergence (v3 mechanism)
    float role_sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    for (int i = 0; i < 4; i++) {
        if (a.role[i] / role_sum > ANTI_CONVERGENCE_THRESH) {
            // Dominant role detected - drift a different role
            int drift_idx = (i + 1 + (a.rng % 3)) % 4;
            a.role[drift_idx] += ANTI_CONVERGENCE_DRIFT;
            a.role[i] -= ANTI_CONVERGENCE_DRIFT;
            break;
        }
    }
    
    // Normalize roles
    float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    for (int i = 0; i < 4; i++) a.role[i] /= sum;
    
    // Calculate behavioral parameters based on roles
    float detect_range = DETECT_RANGE_MIN + a.role[ROLE_EXPLORE] * 
                        (DETECT_RANGE_MAX - DETECT_RANGE_MIN);
    float grab_range = GRAB_RANGE_MIN + a.role[ROLE_COLLECT] * 
                      (GRAB_RANGE_MAX - GRAB_RANGE_MIN);
    
    // Find nearest resource
    float nearest_dist_sq;
    int nearest_res = find_nearest_resource(a.x, a.y, resources, detect_range, nearest_dist_sq);
    
    // v49: Check for pheromones
    float pheromone_attract_x = 0.0f, pheromone_attract_y = 0.0f;
    int strongest_pheromone = find_strongest_pheromone(a.x, a.y, pheromones, a.arch,
                                                      pheromone_attract_x, pheromone_attract_y);
    
    // Determine movement based on roles and pheromones
    float target_x = a.x, target_y = a.y;
    
    if (nearest_res != -1 && nearest_dist_sq < detect_range * detect_range) {
        // Move toward resource
        Resource &r = resources[nearest_res];
        target_x = r.x;
        target_y = r.y;
        
        // Try to collect if in range
        if (nearest_dist_sq < grab_range * grab_range) {
            // Collect resource
            float collect_bonus = 1.0f + a.role[ROLE_COLLECT] * 0.5f;  // Up to 50% bonus
            a.energy += r.value * collect_bonus;
            a.fitness += r.value * collect_bonus;
            r.collected = 1;
            
            // v49: Drop pheromone at resource location
            int pheromone_idx = nearest_res * 2 + a.arch;
            pheromones[pheromone_idx].x = r.x;
            pheromones[pheromone_idx].y = r.y;
            pheromones[pheromone_idx].strength = PHEROMONE_DROP_STRENGTH;
            pheromones[pheromone_idx].arch = a.arch;
        }
    } else if (strongest_pheromone != -1) {
        // v49: Move toward strongest pheromone
        target_x = pheromone_attract_x;
        target_y = pheromone_attract_y;
    }
    
    // Apply movement with pheromone influence
    float dx = target_x - a.x;
    float dy = target_y - a.y;
    float dist = sqrtf(dx*dx + dy*dy + 1e-6f);
    
    // Blend between random walk and directed movement based on roles
    float explore_weight = a.role[ROLE_EXPLORE];
    float directed_weight = (1.0f - explore_weight) * 
                           (strongest_pheromone != -1 ? PHEROMONE_INFLUENCE : 1.0f);
    
    if (dist > 0.001f) {
        a.vx = a.vx * 0.9f + (dx / dist) * 0.001f * directed_weight;
        a.vy = a.vy * 0.9f + (dy / dist) * 0.001f * directed_weight;
    }
    
    // Add random exploration
    a.vx += (lcgf(a.rng) - 0.5f) * 0.0005f * explore_weight;
    a.vy += (lcgf(a.rng) - 0.5f) * 0.0005f * explore_weight;
    
    // Update position with boundary check
    a.x += a.vx;
    a.y += a.vy;
    
    if (a.x < MIN_COORD) { a.x = MIN_COORD; a.vx = -a.vx * 0.5f; }
    if (a.x > MAX_COORD) { a.x = MAX_COORD; a.vx = -a.vx * 0.5f; }
    if (a.y < MIN_COORD) { a.y = MIN_COORD; a.vy = -a.vy * 0.5f; }
    if (a.y > MAX_COORD) { a.y = MAX_COORD; a.vy = -a.vy * 0.5f; }
    
    // Communication role: broadcast resource location to nearby same-arch agents
    if (a.role[ROLE_COMM] > 0.3f && nearest_res != -1) {
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            if (other.arch != a.arch) continue;
            
            float dx2 = other.x - a.x;
            float dy2 = other.y - a.y;
            float d2 = dx2*dx2 + dy2*dy2;
            
            if (d2 < COMM_RANGE * COMM_RANGE) {
                // Influence neighbor's movement toward resource
                Resource &r = resources[nearest_res];
                float influence = a.role[ROLE_COMM] * COUPLING_SAME;
                other.vx += (r.x - other.x) * influence * 0.001f;
                other.vy += (r.y - other.y) * influence * 0.001f;
            }
        }
    }
    
    // Defense role: territory and perturbation resistance
    if (a.role[ROLE_DEFEND] > 0.3f) {
        // Count nearby defenders of same archetype
        int defender_count = 0;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            if (other.arch != a.arch) continue;
            if (other.role[ROLE_DEFEND] < 0.3f) continue;
            
            float dx2 = other.x - a.x;
            float dy2 = other.y - a.y;
            float d2 = dx2*dx2 + dy2*dy2;
            
            if (d2 < TERRITORY_RANGE * TERRITORY_RANGE) {
                defender_count++;
            }
        }
        
        // Apply defense boost
        float boost = 1.0f + defender_count * DEFEND_BOOST_PER_NEIGHBOR * a.role[ROLE_DEFEND];
        a.energy *= boost;
        a.fitness += (boost - 1.0f) * 0.01f;
    }
    
    // Apply random perturbations (every 50 ticks)
    if (tick % 50 == 0 && lcgf(a.rng) < 0.1f) {
        float resist = a.role[ROLE_DEFEND] > 0.3f ? 0.7f : 1.0f;
        a.energy *= (1.0f - PERTURB_ENERGY_LOSS * resist);
        a.vx += (lcgf(a.rng) - 0.5f) * 0.02f;
        a.vy += (lcgf(a.rng) - 0.5f) * 0.02f;
    }
    
    // Accumulate energy sum for this archetype
    atomicAdd(&arch_energy_sum[a.arch], a.energy);
}

// Decay pheromones kernel (v49 novel)
__global__ void decay_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES * 2) return;
    
    pheromones[idx].strength *= PH