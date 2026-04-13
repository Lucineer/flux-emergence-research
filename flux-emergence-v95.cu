/*
CUDA Simulation Experiment v95: STIGMERGY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will improve collective efficiency for specialized agents
            more than for uniform agents, increasing the specialist advantage ratio.
Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence.
Novel: Pheromone trails with exponential decay, visible to all agents within range.
Comparison: Specialized (role[arch]=0.7) vs Uniform (all roles=0.25).
Expected: Specialists should better utilize pheromone information due to role coordination.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;
const float WORLD_SIZE = 1.0f;
const float SPEED = 0.002f;
const float ENERGY_DECAY = 0.999f;
const float PERTURB_PROB = 0.001f;
const float PERTURB_ENERGY_MUL = 0.5f;
const float ANTI_CONV_THRESH = 0.9f;
const float ANTI_CONV_DRIFT = 0.01f;
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;

// Role indices
const int ROLE_EXPLORE = 0;
const int ROLE_COLLECT = 1;
const int ROLE_COMM = 2;
const int ROLE_DEFEND = 3;

// Ranges
const float DETECT_MIN = 0.03f;
const float DETECT_MAX = 0.07f;
const float GRAB_MIN = 0.02f;
const float GRAB_MAX = 0.04f;
const float COMM_RANGE = 0.06f;
const float DEFEND_RANGE = 0.05f;
const float PHEROMONE_RANGE = 0.08f;

// Pheromone constants
const float PHEROMONE_DROP = 1.0f;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_THRESHOLD = 0.01f;

// Agent archetype
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];
    float fitness;
    int arch;
    unsigned int rng;
};

// Resource
struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone trail point
struct Pheromone {
    float x, y;
    float strength;
    int arch;  // which archetype left it
};

// LCG RNG
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize agents
__global__ void init_agents(Agent *agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    a.rng = idx * 123456789 + 12345;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = lcgf(a.rng) * 2.0f - 1.0f;
    a.vy = lcgf(a.rng) * 2.0f - 1.0f;
    float len = sqrtf(a.vx * a.vx + a.vy * a.vy);
    if (len > 0) {
        a.vx = a.vx / len * SPEED;
        a.vy = a.vy / len * SPEED;
    }
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % 2;  // Two archetypes
    
    if (specialized) {
        // Specialized: strong in one role based on arch
        for (int i = 0; i < 4; i++) a.role[i] = 0.1f;
        a.role[a.arch] = 0.7f;
    } else {
        // Uniform: all roles equal
        for (int i = 0; i < 4; i++) a.role[i] = 0.25f;
    }
}

// Initialize resources
__global__ void init_resources(Resource *resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = idx * 987654321 + 67890;
    resources[idx].x = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
    resources[idx].y = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
    resources[idx].value = 0.5f + (lcg(rng) & 0xFFFFFF) / 16777216.0f * 0.5f;
    resources[idx].collected = 0;
}

// Initialize pheromones
__global__ void init_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS * 2) return;  // Max pheromone count
    
    pheromones[idx].strength = 0.0f;
}

// Main simulation tick
__global__ void tick(Agent *agents, Resource *resources, Pheromone *pheromones, 
                     int *pheromone_count, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // 1. Energy decay
    a.energy *= ENERGY_DECAY;
    
    // 2. Random perturbation
    if (lcgf(a.rng) < PERTURB_PROB) {
        float resist = 1.0f - a.role[ROLE_DEFEND] * 0.5f;
        a.energy *= (1.0f - (1.0f - PERTURB_ENERGY_MUL) * resist);
        a.vx = (lcgf(a.rng) * 2.0f - 1.0f) * SPEED;
        a.vy = (lcgf(a.rng) * 2.0f - 1.0f) * SPEED;
    }
    
    // 3. Anti-convergence
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a.role[i] - 0.25f);
    }
    if (similarity > ANTI_CONV_THRESH) {
        int dominant = 0;
        for (int i = 1; i < 4; i++) {
            if (a.role[i] > a.role[dominant]) dominant = i;
        }
        int to_drift = (dominant + 1 + (a.rng % 3)) % 4;
        a.role[to_drift] += ANTI_CONV_DRIFT;
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int i = 0; i < 4; i++) a.role[i] /= sum;
    }
    
    // 4. Movement with pheromone influence
    float best_pheromone_x = 0.0f, best_pheromone_y = 0.0f;
    float best_strength = 0.0f;
    
    // Scan pheromones
    for (int i = 0; i < *pheromone_count; i++) {
        Pheromone &p = pheromones[i];
        if (p.strength < PHEROMONE_THRESHOLD) continue;
        
        float dx = p.x - a.x;
        float dy = p.y - a.y;
        // Wrap-around
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist2 = dx * dx + dy * dy;
        if (dist2 < PHEROMONE_RANGE * PHEROMONE_RANGE) {
            if (p.strength > best_strength) {
                best_strength = p.strength;
                best_pheromone_x = dx;
                best_pheromone_y = dy;
            }
        }
    }
    
    // Blend pheromone attraction with random motion
    if (best_strength > 0) {
        float len = sqrtf(best_pheromone_x * best_pheromone_x + 
                         best_pheromone_y * best_pheromone_y);
        if (len > 0) {
            float influence = a.role[ROLE_EXPLORE] * 0.5f;  // Explorers use pheromones more
            a.vx = a.vx * (1.0f - influence) + 
                   (best_pheromone_x / len * SPEED) * influence;
            a.vy = a.vy * (1.0f - influence) + 
                   (best_pheromone_y / len * SPEED) * influence;
        }
    }
    
    // Normalize velocity
    float len = sqrtf(a.vx * a.vx + a.vy * a.vy);
    if (len > 0) {
        a.vx = a.vx / len * SPEED;
        a.vy = a.vy / len * SPEED;
    }
    
    // 5. Position update with wrap-around
    a.x += a.vx;
    a.y += a.vy;
    if (a.x < 0) a.x += 1.0f;
    if (a.x >= 1.0f) a.x -= 1.0f;
    if (a.y < 0) a.y += 1.0f;
    if (a.y >= 1.0f) a.y -= 1.0f;
    
    // 6. Resource interaction
    float detect_range = DETECT_MIN + a.role[ROLE_EXPLORE] * (DETECT_MAX - DETECT_MIN);
    float grab_range = GRAB_MIN + a.role[ROLE_COLLECT] * (GRAB_MAX - GRAB_MIN);
    
    float best_dx = 0.0f, best_dy = 0.0f;
    float best_value = 0.0f;
    int best_idx = -1;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist2 = dx * dx + dy * dy;
        if (dist2 < detect_range * detect_range) {
            float value = r.value * (1.0f + a.role[ROLE_COLLECT] * 0.5f);  // Collector bonus
            if (value > best_value) {
                best_value = value;
                best_dx = dx;
                best_dy = dy;
                best_idx = i;
            }
        }
    }
    
    // 7. Collect resource if in grab range
    if (best_idx >= 0) {
        float dist = sqrtf(best_dx * best_dx + best_dy * best_dy);
        if (dist < grab_range) {
            Resource &r = resources[best_idx];
            
            // Territory bonus from nearby defenders of same arch
            float territory_bonus = 1.0f;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent &other = agents[j];
                if (other.arch != a.arch) continue;
                
                float dx2 = other.x - a.x;
                float dy2 = other.y - a.y;
                if (dx2 > 0.5f) dx2 -= 1.0f;
                if (dx2 < -0.5f) dx2 += 1.0f;
                if (dy2 > 0.5f) dy2 -= 1.0f;
                if (dy2 < -0.5f) dy2 += 1.0f;
                
                float dist2 = dx2 * dx2 + dy2 * dy2;
                if (dist2 < DEFEND_RANGE * DEFEND_RANGE) {
                    territory_bonus += other.role[ROLE_DEFEND] * 0.2f;
                }
            }
            
            float gain = r.value * (1.0f + a.role[ROLE_COLLECT] * 0.5f) * territory_bonus;
            a.energy += gain;
            a.fitness += gain;
            r.collected = 1;
            
            // Drop pheromone at resource location
            int p_idx = atomicAdd(pheromone_count, 1);
            if (p_idx < AGENTS * 2) {
                pheromones[p_idx].x = r.x;
                pheromones[p_idx].y = r.y;
                pheromones[p_idx].strength = PHEROMONE_DROP;
                pheromones[p_idx].arch = a.arch;
            }
        }
    }
    
    // 8. Communication
    if (a.role[ROLE_COMM] > 0.3f) {
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx > 0.5f) dx -= 1.0f;
            if (dx < -0.5f) dx += 1.0f;
            if (dy > 0.5f) dy -= 1.0f;
            if (dy < -0.5f) dy += 1.0f;
            
            float dist2 = dx * dx + dy * dy;
            if (dist2 < COMM_RANGE * COMM_RANGE) {
                // Coupling: influence roles
                float coupling = (a.arch == other.arch) ? COUPLING_SAME : COUPLING_DIFF;
                for (int j = 0; j < 4; j++) {
                    float diff = a.role[j] - other.role[j];
                    other.role[j] += diff * coupling * a.role[ROLE_COMM];
                }
                // Renormalize other's roles
                float sum = other.role[0] + other.role[1] + other.role[2] + other.role[3];
                for (int j = 0; j < 4; j++) other.role[j] /= sum;
            }
        }
    }
}

// Decay pheromones
__global__ void decay_pheromones(Pheromone *pheromones, int *pheromone_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *pheromone_count) return;
    
    pheromones[idx].strength *= PHEROMONE_DECAY;
}

// Reset resources periodically
__global__ void reset_resources(Resource *resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    // Respawn every 50 ticks
    if (tick_num % 50 == 0) {
        unsigned int rng = idx * 135791113 + tick_num;
        resources[idx].x = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
        resources[idx].y = (lcg(rng) & 0xFFFFFF) / 16777216.0f;
        resources[idx].value = 0.5f + (lcg(rng) & 0xFFFFFF) / 16777216.0f * 0.5f;
        resources[idx].collected = 0;
    }
}

int main() {
    // Allocate host memory
    Agent *h_agents_spec, *h_agents_uniform;
    Resource *h_resources;
    Pheromone *h_pheromones_spec, *h_pheromones_uniform;
    
    h_agents_spec = (Agent*)malloc(AGENTS * sizeof(Agent));
    h_agents_uniform = (Agent*)malloc(AGENTS * sizeof(Agent));
    h_resources = (Resource*)malloc(RESOURCES * sizeof(Resource));
    h_pheromones_spec = (Pheromone*)malloc(AGENTS * 2 * sizeof(Pheromone));
    h_pheromones_uniform = (P