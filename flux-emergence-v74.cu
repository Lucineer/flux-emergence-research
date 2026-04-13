
/*
CUDA Simulation Experiment v74: Stigmergy with Pheromone Trails
Testing: Whether pheromone trails left at resource locations improve specialist efficiency
Prediction: Pheromones will help uniform agents more than specialists (reducing specialist advantage)
  because specialists already have optimized roles, while uniform agents benefit from shared information
Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
Novel: Agents leave pheromone markers at collected resources that decay over time
  - Pheromone intensity: 1.0 at drop, decays 0.97/tick
  - Detection: All agents sense pheromones within 0.08 range
  - Effect: Move toward strongest pheromone if no resource detected
  - Specialist archetypes: explore=0.7,0.1,0.1,0.1; collect=0.1,0.7,0.1,0.1; comm=0.1,0.1,0.7,0.1; defend=0.1,0.1,0.1,0.7
  - Uniform control: all roles=0.25
Expected: Specialist advantage ratio < 1.61x (v8 baseline)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define AGENTS 1024
#define RESOURCES 128
#define TICKS 500
#define ARCHETYPES 4
#define PHEROMONE_GRID_SIZE 256
#define PHEROMONE_DECAY 0.97f

// Linear congruential generator
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

struct Resource {
    float x, y;
    float value;
    bool collected;
    int last_collected_tick;
};

struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];  // explore, collect, communicate, defend
    float fitness;
    int arch;  // 0=explorer,1=collector,2=communicator,3=defender
    unsigned int rng;
};

struct Pheromone {
    float intensity;
    float target_x, target_y;
};

__device__ float distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx*dx + dy*dy);
}

__global__ void init_agents(Agent *agents, Pheromone *pheromones, int tick) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    unsigned int seed = idx * 17 + tick * 7919;
    agents[idx].x = lcgf(seed) * 2.0f - 1.0f;
    agents[idx].y = lcgf(seed) * 2.0f - 1.0f;
    agents[idx].vx = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].rng = idx * 12345 + 6789;
    
    // Specialist vs uniform groups
    if (idx < AGENTS/2) {  // Specialists
        agents[idx].arch = idx % ARCHETYPES;
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = (i == agents[idx].arch) ? 0.7f : 0.1f;
        }
    } else {  // Uniform control
        agents[idx].arch = -1;
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

__global__ void init_resources(Resource *resources, int tick) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int seed = idx * 13 + tick * 9973;
    resources[idx].x = lcgf(seed) * 2.0f - 1.0f;
    resources[idx].y = lcgf(seed) * 2.0f - 1.0f;
    resources[idx].value = 0.8f + lcgf(seed) * 0.4f;
    resources[idx].collected = false;
    resources[idx].last_collected_tick = -1000;
}

__global__ void init_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE) return;
    pheromones[idx].intensity = 0.0f;
    pheromones[idx].target_x = 0.0f;
    pheromones[idx].target_y = 0.0f;
}

__device__ int pheromone_index(float x, float y) {
    int ix = (int)((x + 1.0f) * 0.5f * PHEROMONE_GRID_SIZE);
    int iy = (int)((y + 1.0f) * 0.5f * PHEROMONE_GRID_SIZE);
    ix = max(0, min(PHEROMONE_GRID_SIZE-1, ix));
    iy = max(0, min(PHEROMONE_GRID_SIZE-1, iy));
    return iy * PHEROMONE_GRID_SIZE + ix;
}

__global__ void decay_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE) return;
    pheromones[idx].intensity *= PHEROMONE_DECAY;
    if (pheromones[idx].intensity < 0.001f) {
        pheromones[idx].intensity = 0.0f;
    }
}

__global__ void tick(Agent *agents, Resource *resources, Pheromone *pheromones, 
                     int current_tick, float *specialist_fitness, float *uniform_fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = (idx + 37) % AGENTS;
    Agent &other = agents[other_idx];
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a.role[i] - other.role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant role
        int drift_role;
        do {
            drift_role = (int)(lcgf(a.rng) * 4);
        } while (drift_role == a.arch || a.arch == -1);
        
        float drift = lcgf(a.rng) * 0.02f - 0.01f;
        a.role[drift_role] += drift;
        a.role[drift_role] = max(0.0f, min(1.0f, a.role[drift_role]));
        
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int i = 0; i < 4; i++) a.role[i] /= sum;
    }
    
    // Movement with role-based behavior
    float explore_range = 0.03f + a.role[0] * 0.04f;
    float collect_range = 0.02f + a.role[1] * 0.02f;
    float comm_range = 0.04f + a.role[2] * 0.02f;
    float defend_range = 0.03f + a.role[3] * 0.02f;
    
    // Find nearest resource
    float nearest_dist = 100.0f;
    float nearest_x = 0.0f, nearest_y = 0.0f;
    bool found_resource = false;
    
    for (int i = 0; i < RESOURCES; i++) {
        if (resources[i].collected) continue;
        
        float d = distance(a.x, a.y, resources[i].x, resources[i].y);
        if (d < explore_range && d < nearest_dist) {
            nearest_dist = d;
            nearest_x = resources[i].x;
            nearest_y = resources[i].y;
            found_resource = true;
        }
    }
    
    // Check pheromones if no resource found
    float best_pheromone = 0.0f;
    float pheromone_x = 0.0f, pheromone_y = 0.0f;
    
    if (!found_resource) {
        int base_idx = pheromone_index(a.x, a.y);
        int search_radius = 2;
        
        for (int dy = -search_radius; dy <= search_radius; dy++) {
            for (int dx = -search_radius; dx <= search_radius; dx++) {
                int check_idx = base_idx + dy * PHEROMONE_GRID_SIZE + dx;
                if (check_idx >= 0 && check_idx < PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE) {
                    if (pheromones[check_idx].intensity > best_pheromone) {
                        best_pheromone = pheromones[check_idx].intensity;
                        pheromone_x = pheromones[check_idx].target_x;
                        pheromone_y = pheromones[check_idx].target_y;
                    }
                }
            }
        }
    }
    
    // Movement decision
    float target_x = 0.0f, target_y = 0.0f;
    if (found_resource) {
        target_x = nearest_x;
        target_y = nearest_y;
    } else if (best_pheromone > 0.1f) {
        target_x = pheromone_x;
        target_y = pheromone_y;
    } else {
        // Random walk
        target_x = a.x + lcgf(a.rng) * 0.1f - 0.05f;
        target_y = a.y + lcgf(a.rng) * 0.1f - 0.05f;
    }
    
    // Move toward target
    float dx = target_x - a.x;
    float dy = target_y - a.y;
    float dist_to_target = sqrtf(dx*dx + dy*dy);
    if (dist_to_target > 0.001f) {
        a.vx = dx / dist_to_target * 0.01f;
        a.vy = dy / dist_to_target * 0.01f;
    }
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // Boundary check
    if (a.x < -1.0f) { a.x = -1.0f; a.vx = -a.vx; }
    if (a.x > 1.0f) { a.x = 1.0f; a.vx = -a.vx; }
    if (a.y < -1.0f) { a.y = -1.0f; a.vy = -a.vy; }
    if (a.y > 1.0f) { a.y = 1.0f; a.vy = -a.vy; }
    
    // Resource collection
    if (found_resource && nearest_dist < collect_range) {
        for (int i = 0; i < RESOURCES; i++) {
            if (resources[i].collected) continue;
            
            float d = distance(a.x, a.y, resources[i].x, resources[i].y);
            if (d < collect_range) {
                // Collection bonus based on collect role
                float bonus = 1.0f + a.role[1] * 0.5f;
                float gained = resources[i].value * bonus;
                
                // Territory bonus from nearby defenders of same archetype
                int nearby_defenders = 0;
                for (int j = 0; j < AGENTS; j++) {
                    if (j == idx) continue;
                    Agent &other = agents[j];
                    if (other.arch == a.arch && other.arch != -1) {
                        float dist = distance(a.x, a.y, other.x, other.y);
                        if (dist < defend_range && other.role[3] > 0.5f) {
                            nearby_defenders++;
                        }
                    }
                }
                gained *= (1.0f + nearby_defenders * 0.2f);
                
                a.energy += gained;
                a.fitness += gained;
                resources[i].collected = true;
                resources[i].last_collected_tick = current_tick;
                
                // Leave pheromone at collection site
                int pidx = pheromone_index(resources[i].x, resources[i].y);
                pheromones[pidx].intensity = 1.0f;
                pheromones[pidx].target_x = resources[i].x;
                pheromones[pidx].target_y = resources[i].y;
                
                break;
            }
        }
    }
    
    // Communication
    if (a.role[2] > 0.3f) {
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            float d = distance(a.x, a.y, other.x, other.y);
            if (d < comm_range) {
                // Share resource location if known
                if (found_resource) {
                    // Influence other's movement slightly
                    float influence = a.role[2] * 0.3f;
                    other.vx += (nearest_x - other.x) * influence * 0.01f;
                    other.vy += (nearest_y - other.y) * influence * 0.01f;
                }
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (current_tick % 50 == 25) {
        // Defenders resist perturbation
        float resistance = a.role[3] * 0.8f;
        if (lcgf(a.rng) > resistance) {
            a.energy *= 0.5f;
            a.vx += lcgf(a.rng) * 0.02f - 0.01f;
            a.vy += lcgf(a.rng) * 0.02f - 0.01f;
        }
    }
    
    // Resource respawn (every 50 ticks at different time than perturbation)
    if (current_tick % 50 == 0 && current_tick > 0) {
        for (int i = 0; i < RESOURCES; i++) {
            if (resources[i].collected && (current_tick - resources[i].last_collected_tick) > 20) {
                resources[i].collected = false;
                resources[i].value = 0.8f + lcgf(a.rng) * 0.4f;
            }
        }
    }
    
    // Record fitness for this tick
    if (idx < AGENTS/2) {
        atomicAdd(specialist_fitness, a.fitness);
    } else {
        atomicAdd(uniform_fitness, a.fitness);
    }
}

int main() {
    // Allocate memory
    Agent *d_agents;
    Resource *d_resources;
    Pheromone *d_pheromones;
    float *d_specialist_fitness, *d_uniform_fitness;
    float h_specialist_fitness = 0.0f, h_uniform_fitness = 0.0f;
    
    cudaMalloc(&d_agents, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    cudaMalloc(&d_pheromones, sizeof(Pheromone) * PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE);
    cudaMalloc(&d_specialist_fitness, sizeof(float));
    cudaMalloc(&d_uniform_fitness, sizeof(float));
    
    // Initialize
    dim3 block(256);
    dim3 grid_agents((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE + 255) / 256);
    
    init_agents<<<grid_agents, block>>>(d_agents, d_pheromones, 0);
    init_resources<<<grid_res, block>>>(d_resources, 0);
    init_pheromones<<<grid_ph, block>>>(d_pheromones);
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int tick_num = 0; tick_num < TICKS; tick_num++) {
        cudaM