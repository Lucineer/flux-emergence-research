
/*
CUDA Simulation Experiment v44: STIGMERGY TRAILS
Testing: Pheromone trails at resource locations that decay over time.
Prediction: Specialists will use trails more efficiently, increasing their advantage.
Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence.
Comparison: Specialized (role[arch]=0.7) vs Uniform (all roles=0.25).
Expected: Specialists show >1.61x advantage due to trail-following efficiency.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK = 256;
const float WORLD_SIZE = 1.0f;
const float SPEED = 0.002f;
const float ENERGY_DECAY = 0.999f;
const float PERTURB_PROB = 0.001f;
const float PERTURB_ENERGY_MUL = 0.5f;
const float ANTI_CONV_THRESH = 0.9f;
const float ANTI_CONV_DRIFT = 0.01f;
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;
const float DEFEND_BOOST = 0.2f;

// Stigmergy constants
const float PHEROMONE_DROP = 0.3f;
const float PHEROMONE_DECAY = 0.97f;
const float TRAIL_DETECTION_RANGE = 0.08f;
const float TRAIL_FOLLOW_STRENGTH = 0.4f;

// Role indices
enum { ROLE_EXPLORE, ROLE_COLLECT, ROLE_COMM, ROLE_DEFEND };

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];
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

// Pheromone struct for stigmergy
struct Pheromone {
    float x, y;
    float strength;
    int arch;
};

// LCG RNG
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents
__global__ void init_agents(Agent *agents, int seed, bool specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    a.rng = seed + idx * 137;
    a.x = lcgf(a.rng) * WORLD_SIZE;
    a.y = lcgf(a.rng) * WORLD_SIZE;
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
        // Specialized: high in own archetype's role
        for (int i = 0; i < 4; i++) a.role[i] = 0.1f;
        a.role[a.arch] = 0.7f;
    } else {
        // Uniform control
        for (int i = 0; i < 4; i++) a.role[i] = 0.25f;
    }
}

// Initialize resources
__global__ void init_resources(Resource *resources, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = resources[idx];
    unsigned int rng = seed + idx * 7919;
    r.x = (lcg(rng) / 4294967296.0f) * WORLD_SIZE;
    r.y = (lcg(rng) / 4294967296.0f) * WORLD_SIZE;
    r.value = 0.8f + (lcg(rng) / 4294967296.0f) * 0.4f;
    r.collected = 0;
}

// Initialize pheromones
__global__ void init_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Pheromone &p = pheromones[idx];
    p.strength = 0.0f;
    p.arch = -1;
}

// Update pheromones (decay and remove weak ones)
__global__ void update_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Pheromone &p = pheromones[idx];
    p.strength *= PHEROMONE_DECAY;
    if (p.strength < 0.01f) {
        p.strength = 0.0f;
        p.arch = -1;
    }
}

// Main simulation tick
__global__ void tick(Agent *agents, Resource *resources, Pheromone *pheromones, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= ENERGY_DECAY;
    
    // Perturbation
    if (lcgf(a.rng) < PERTURB_PROB) {
        float resist = 1.0f - a.role[ROLE_DEFEND] * 0.8f;
        a.energy *= (1.0f - (1.0f - PERTURB_ENERGY_MUL) * resist);
        a.vx += (lcgf(a.rng) * 2.0f - 1.0f) * 0.01f;
        a.vy += (lcgf(a.rng) * 2.0f - 1.0f) * 0.01f;
    }
    
    // Anti-convergence
    float max_role = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < 4; i++) {
        if (a.role[i] > max_role) {
            max_role = a.role[i];
            max_idx = i;
        }
    }
    if (max_role > ANTI_CONV_THRESH) {
        int drift_idx;
        do {
            drift_idx = lcg(a.rng) % 4;
        } while (drift_idx == max_idx);
        a.role[drift_idx] += ANTI_CONV_DRIFT;
        a.role[max_idx] -= ANTI_CONV_DRIFT;
    }
    
    // Coupling with nearby agents
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent &other = agents[i];
        float dx = other.x - a.x;
        float dy = other.y - a.y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 < 0.04f && dist2 > 0.000001f) {
            float dist = sqrtf(dist2);
            float coupling = (a.arch == other.arch) ? COUPLING_SAME : COUPLING_DIFF;
            coupling /= dist * 10.0f;
            for (int r = 0; r < 4; r++) {
                float diff = other.role[r] - a.role[r];
                a.role[r] += diff * coupling;
                other.role[r] -= diff * coupling;
            }
        }
    }
    
    // Normalize roles
    float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    for (int i = 0; i < 4; i++) a.role[i] /= sum;
    
    // Role-based behavior
    float explore = a.role[ROLE_EXPLORE];
    float collect = a.role[ROLE_COLLECT];
    float comm = a.role[ROLE_COMM];
    float defend = a.role[ROLE_DEFEND];
    
    // Detection and collection ranges
    float detect_range = 0.03f + explore * 0.04f;
    float grab_range = 0.02f + collect * 0.02f;
    float comm_range = 0.06f;
    
    // Find nearest resource
    int nearest_idx = -1;
    float nearest_dist2 = 1e6f;
    float nearest_x = 0.0f, nearest_y = 0.0f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 < nearest_dist2) {
            nearest_dist2 = dist2;
            nearest_idx = i;
            nearest_x = r.x;
            nearest_y = r.y;
        }
    }
    
    // STIGMERGY: Follow pheromone trails
    float trail_vx = 0.0f, trail_vy = 0.0f;
    int trail_count = 0;
    for (int i = 0; i < RESOURCES; i++) {
        Pheromone &p = pheromones[i];
        if (p.strength > 0.1f && p.arch == a.arch) {
            float dx = p.x - a.x;
            float dy = p.y - a.y;
            float dist2 = dx * dx + dy * dy;
            if (dist2 < TRAIL_DETECTION_RANGE * TRAIL_DETECTION_RANGE) {
                float w = p.strength / (sqrtf(dist2) + 0.01f);
                trail_vx += dx * w;
                trail_vy += dy * w;
                trail_count++;
            }
        }
    }
    if (trail_count > 0) {
        float len = sqrtf(trail_vx * trail_vx + trail_vy * trail_vy);
        if (len > 0) {
            trail_vx = trail_vx / len * SPEED;
            trail_vy = trail_vy / len * SPEED;
            a.vx = a.vx * (1.0f - TRAIL_FOLLOW_STRENGTH) + trail_vx * TRAIL_FOLLOW_STRENGTH;
            a.vy = a.vy * (1.0f - TRAIL_FOLLOW_STRENGTH) + trail_vy * TRAIL_FOLLOW_STRENGTH;
        }
    }
    
    // Communication: broadcast nearest resource location
    if (comm > 0.1f && nearest_idx >= 0 && nearest_dist2 < detect_range * detect_range) {
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            if (other.arch != a.arch) continue;
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx * dx + dy * dy < comm_range * comm_range) {
                // Influence other's velocity toward resource
                float inf = comm * 0.3f;
                float rdx = nearest_x - other.x;
                float rdy = nearest_y - other.y;
                float rlen = sqrtf(rdx * rdx + rdy * rdy);
                if (rlen > 0) {
                    other.vx = other.vx * (1.0f - inf) + (rdx / rlen * SPEED) * inf;
                    other.vy = other.vy * (1.0f - inf) + (rdy / rlen * SPEED) * inf;
                }
            }
        }
    }
    
    // Move toward resources if detected
    if (nearest_idx >= 0 && nearest_dist2 < detect_range * detect_range) {
        float dx = nearest_x - a.x;
        float dy = nearest_y - a.y;
        float dist = sqrtf(nearest_dist2);
        if (dist > 0) {
            a.vx = a.vx * (1.0f - explore) + (dx / dist * SPEED) * explore;
            a.vy = a.vy * (1.0f - explore) + (dy / dist * SPEED) * explore;
        }
        
        // Collect if in range
        if (nearest_dist2 < grab_range * grab_range) {
            Resource &r = resources[nearest_idx];
            if (!r.collected) {
                // Territory boost from nearby defenders
                float defend_boost = 1.0f;
                for (int i = 0; i < AGENTS; i++) {
                    if (i == idx) continue;
                    Agent &other = agents[i];
                    if (other.arch != a.arch) continue;
                    float dx = other.x - a.x;
                    float dy = other.y - a.y;
                    if (dx * dx + dy * dy < 0.04f && other.role[ROLE_DEFEND] > 0.3f) {
                        defend_boost += DEFEND_BOOST;
                    }
                }
                
                float gain = r.value * (1.0f + collect * 0.5f) * defend_boost;
                a.energy += gain;
                a.fitness += gain;
                r.collected = 1;
                
                // STIGMERGY: Drop pheromone at collected resource location
                for (int i = 0; i < RESOURCES; i++) {
                    Pheromone &p = pheromones[i];
                    if (p.strength < 0.01f || p.arch == -1) {
                        p.x = r.x;
                        p.y = r.y;
                        p.strength = PHEROMONE_DROP;
                        p.arch = a.arch;
                        break;
                    }
                }
            }
        }
    }
    
    // Random movement component
    a.vx += (lcgf(a.rng) * 2.0f - 1.0f) * 0.001f;
    a.vy += (lcgf(a.rng) * 2.0f - 1.0f) * 0.001f;
    
    // Velocity limiting
    float vlen = sqrtf(a.vx * a.vx + a.vy * a.vy);
    if (vlen > SPEED * 2.0f) {
        a.vx = a.vx / vlen * SPEED * 2.0f;
        a.vy = a.vy / vlen * SPEED * 2.0f;
    }
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World wrap
    if (a.x < 0) a.x += WORLD_SIZE;
    if (a.x >= WORLD_SIZE) a.x -= WORLD_SIZE;
    if (a.y < 0) a.y += WORLD_SIZE;
    if (a.y >= WORLD_SIZE) a.y -= WORLD_SIZE;
    
    // Respawn resources periodically
    if (tick_num % 50 == 0) {
        for (int i = 0; i < RESOURCES; i++) {
            Resource &r = resources[i];
            if (r.collected) {
                r.x = lcgf(a.rng) * WORLD_SIZE;
                r.y = lcgf(a.rng) * WORLD_SIZE;
                r.value = 0.8f + lcgf(a.rng) * 0.4f;
                r.collected = 0;
            }
        }
    }
}

int main() {
    // Allocate memory
    Agent *d_agents_spec, *d_agents_uniform;
    Resource *d_resources;
    Pheromone *d_pheromones_spec, *d_pheromones_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, RESOURCES * sizeof(Pheromone));
    cudaMalloc(&d_pheromones_uniform, RESOURCES * sizeof(Pheromone));
    
    Agent *h_agents_spec = new Agent[AGENTS];
    Agent *h_agents_uniform = new Agent[AGENTS];
    
    // Initialize
    dim3 block(BLOCK);
    dim3 grid_spec((AGENTS + BLOCK - 1) / BLOCK);
    dim3 grid_res((RESOURCES + BLOCK - 1) / BLOCK);
    
    init_agents<<<grid_spec, block>>>(d_agents_spec, 12345, true);
    init_agents<<<grid_spec, block>>>(d_agents_uniform, 67890, false);
    init_resources