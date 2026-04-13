
/*
CUDA Simulation Experiment v91: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at collected resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents.
Baseline: v8 mechanisms (scarcity, territory, communication) included.
Novelty: Pheromone trails that agents can detect (explorers detect best, collectors follow to resources).
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define AGENTS 1024
#define RESOURCES 128
#define TICKS 500
#define PHEROMONES 512  // Max active pheromone markers

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1103515245 + 12345;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return lcg(state) / 4294967296.0f;
}

// Pheromone marker
struct Pheromone {
    float x, y;
    float strength;
    int arch;  // Which archetype left it
    int age;
};

// Resource
struct Resource {
    float x, y;
    float value;
    bool collected;
};

// Agent
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    char role[4];  // 0:explore, 1:collect, 2:communicate, 3:defend
    float fitness;
    int arch;  // 0:uniform, 1:specialized
    unsigned int rng;
};

// Global device arrays
__device__ Agent d_agents[AGENTS];
__device__ Resource d_resources[RESOURCES];
__device__ Pheromone d_pheromones[PHEROMONES];
__device__ int d_pheromone_count = 0;
__device__ float d_specialist_energy = 0.0f;
__device__ float d_uniform_energy = 0.0f;
__device__ int d_specialist_collected = 0;
__device__ int d_uniform_collected = 0;

// Initialize agents and resources
__global__ void init_simulation(unsigned int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS + RESOURCES) return;
    
    unsigned int rng = seed + idx * 137;
    
    if (idx < AGENTS) {
        // Initialize agent
        d_agents[idx].x = lcgf(rng);
        d_agents[idx].y = lcgf(rng);
        d_agents[idx].vx = lcgf(rng) * 0.02f - 0.01f;
        d_agents[idx].vy = lcgf(rng) * 0.02f - 0.01f;
        d_agents[idx].energy = 1.0f;
        d_agents[idx].fitness = 0.0f;
        d_agents[idx].rng = rng;
        d_agents[idx].arch = (idx < AGENTS/2) ? 0 : 1;  // First half uniform, second half specialized
        
        // Set roles based on archetype
        if (d_agents[idx].arch == 0) {
            // Uniform: all roles equal
            d_agents[idx].role[0] = 0.25f;
            d_agents[idx].role[1] = 0.25f;
            d_agents[idx].role[2] = 0.25f;
            d_agents[idx].role[3] = 0.25f;
        } else {
            // Specialized: strong in one role based on hash
            int role_idx = idx % 4;
            for (int i = 0; i < 4; i++) {
                d_agents[idx].role[i] = (i == role_idx) ? 0.7f : 0.1f;
            }
        }
    } else if (idx < AGENTS + RESOURCES) {
        // Initialize resource
        int res_idx = idx - AGENTS;
        d_resources[res_idx].x = lcgf(rng);
        d_resources[res_idx].y = lcgf(rng);
        d_resources[res_idx].value = 0.8f + lcgf(rng) * 0.4f;
        d_resources[res_idx].collected = false;
    }
}

// Find nearest resource
__device__ int find_nearest_resource(float x, float y, int exclude_idx = -1) {
    int best = -1;
    float best_dist = 1e6;
    for (int i = 0; i < RESOURCES; i++) {
        if (d_resources[i].collected) continue;
        if (i == exclude_idx) continue;
        float dx = d_resources[i].x - x;
        float dy = d_resources[i].y - y;
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

// Find strongest pheromone of same archetype
__device__ int find_best_pheromone(float x, float y, int arch) {
    int best = -1;
    float best_strength = 0.0f;
    for (int i = 0; i < d_pheromone_count; i++) {
        if (d_pheromones[i].arch != arch) continue;
        float dx = d_pheromones[i].x - x;
        float dy = d_pheromones[i].y - y;
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 0.3f && d_pheromones[i].strength > best_strength) {
            best_strength = d_pheromones[i].strength;
            best = i;
        }
    }
    return best;
}

// Main simulation kernel
__global__ void tick(int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent &a = d_agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: if roles too similar, add drift
    float role_sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    float role_norm[4];
    for (int i = 0; i < 4; i++) role_norm[i] = a.role[i] / role_sum;
    
    float max_role = fmaxf(fmaxf(role_norm[0], role_norm[1]), 
                          fmaxf(role_norm[2], role_norm[3]));
    if (max_role > 0.9f) {
        // Find non-dominant role and add drift
        for (int i = 0; i < 4; i++) {
            if (role_norm[i] < 0.2f) {
                a.role[i] += 0.01f;
                break;
            }
        }
    }
    
    // Movement with pheromone influence
    float target_x = 0.0f, target_y = 0.0f;
    int pheromone_target = -1;
    
    // Explorers detect pheromones best
    if (a.role[0] > 0.3f) {
        pheromone_target = find_best_pheromone(a.x, a.y, a.arch);
        if (pheromone_target >= 0) {
            target_x = d_pheromones[pheromone_target].x;
            target_y = d_pheromones[pheromone_target].y;
        }
    }
    
    // Collectors follow pheromones to resources
    if (pheromone_target < 0 && a.role[1] > 0.3f) {
        pheromone_target = find_best_pheromone(a.x, a.y, a.arch);
        if (pheromone_target >= 0) {
            target_x = d_pheromones[pheromone_target].x;
            target_y = d_pheromones[pheromone_target].y;
        }
    }
    
    if (pheromone_target >= 0) {
        // Move toward pheromone
        float dx = target_x - a.x;
        float dy = target_y - a.y;
        float dist = sqrtf(dx*dx + dy*dy) + 1e-6f;
        a.vx = dx / dist * 0.015f;
        a.vy = dy / dist * 0.015f;
    } else {
        // Random walk with role bias
        a.vx += (lcgf(a.rng) - 0.5f) * 0.01f;
        a.vy += (lcgf(a.rng) - 0.5f) * 0.01f;
        
        // Explorers move more
        if (a.role[0] > 0.3f) {
            a.vx *= 1.5f;
            a.vy *= 1.5f;
        }
    }
    
    // Velocity limits
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.03f) {
        a.vx *= 0.03f / speed;
        a.vy *= 0.03f / speed;
    }
    
    // Update position (toroidal world)
    a.x += a.vx;
    a.y += a.vy;
    if (a.x < 0) a.x += 1.0f;
    if (a.x > 1) a.x -= 1.0f;
    if (a.y < 0) a.y += 1.0f;
    if (a.y > 1) a.y -= 1.0f;
    
    // Resource detection and collection
    int nearest_res = find_nearest_resource(a.x, a.y);
    if (nearest_res >= 0) {
        Resource &r = d_resources[nearest_res];
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        // Detection range based on explore role
        float detect_range = 0.03f + a.role[0] * 0.04f;
        
        if (dist < detect_range) {
            // Grab range based on collect role
            float grab_range = 0.02f + a.role[1] * 0.02f;
            
            if (dist < grab_range && !r.collected) {
                // Collect resource
                float value = r.value;
                
                // Collector bonus
                if (a.role[1] > 0.3f) value *= 1.5f;
                
                // Territory bonus from nearby defenders of same archetype
                int defenders_nearby = 0;
                for (int i = 0; i < AGENTS; i++) {
                    if (i == idx) continue;
                    Agent &other = d_agents[i];
                    if (other.arch != a.arch) continue;
                    if (other.role[3] < 0.3f) continue;
                    float odx = other.x - a.x;
                    float ody = other.y - a.y;
                    if (sqrtf(odx*odx + ody*ody) < 0.1f) {
                        defenders_nearby++;
                    }
                }
                value *= 1.0f + defenders_nearby * 0.2f;
                
                a.energy += value;
                a.fitness += value;
                r.collected = true;
                
                // Leave pheromone at collected location
                int p_idx = atomicAdd(&d_pheromone_count, 1);
                if (p_idx < PHEROMONES) {
                    d_pheromones[p_idx].x = r.x;
                    d_pheromones[p_idx].y = r.y;
                    d_pheromones[p_idx].strength = 1.0f;
                    d_pheromones[p_idx].arch = a.arch;
                    d_pheromones[p_idx].age = 0;
                }
                
                // Track statistics
                if (a.arch == 0) {
                    atomicAdd(&d_uniform_energy, value);
                    atomicAdd(&d_uniform_collected, 1);
                } else {
                    atomicAdd(&d_specialist_energy, value);
                    atomicAdd(&d_specialist_collected, 1);
                }
            }
            
            // Communication
            if (a.role[2] > 0.3f) {
                // Broadcast location to nearby agents of same archetype
                for (int i = 0; i < AGENTS; i++) {
                    if (i == idx) continue;
                    Agent &other = d_agents[i];
                    if (other.arch != a.arch) continue;
                    float odx = other.x - a.x;
                    float ody = other.y - a.y;
                    if (sqrtf(odx*odx + ody*ody) < 0.06f) {
                        // Influence neighbor's movement toward resource
                        float influence = a.role[2] * 0.5f;
                        other.vx += (r.x - other.x) * influence * 0.01f;
                        other.vy += (r.y - other.y) * influence * 0.01f;
                    }
                }
            }
        }
    }
    
    // Perturbation every 50 ticks
    if (tick_num % 50 == 25) {
        // Defenders resist perturbation
        if (a.role[3] < 0.5f || lcgf(a.rng) > 0.7f) {
            a.energy *= 0.5f;
            a.vx = lcgf(a.rng) * 0.02f - 0.01f;
            a.vy = lcgf(a.rng) * 0.02f - 0.01f;
        }
    }
}

// Update and decay pheromones
__global__ void update_pheromones() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= d_pheromone_count) return;
    
    d_pheromones[idx].age++;
    d_pheromones[idx].strength *= 0.95f;  // Decay
    
    // Remove old pheromones by swapping with last
    if (d_pheromones[idx].strength < 0.01f || d_pheromones[idx].age > 100) {
        int last = d_pheromone_count - 1;
        if (idx < last) {
            d_pheromones[idx] = d_pheromones[last];
        }
        atomicSub(&d_pheromone_count, 1);
    }
}

// Reset resources periodically
__global__ void reset_resources() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    // Every 50 ticks, respawn some resources
    if (d_resources[idx].collected) {
        unsigned int rng = idx * 7919;
        if (lcgf(rng) < 0.3f) {  // 30% respawn chance
            d_resources[idx].x = lcgf(rng);
            d_resources[idx].y = lcgf(rng);
            d_resources[idx].value = 0.8f + lcgf(rng) * 0.4f;
            d_resources[idx].collected = false;
        }
    }
}

int main() {
    printf("Experiment v91: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone markers enhance specialist coordination\n");
    printf("Prediction: Specialists will outperform uniform by >1.61x with pheromones\n");
    printf("Agents: %d (512 uniform, 512 specialized)\n", AGENTS);
    printf("Resources: %d (scarce)\n", RESOURCES);
    printf("Ticks: %d\n\n", TICKS);
    
    // Initialize
    init_simulation<<<1, AGENTS + RESOURCES>>>(time(NULL));
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        tick<<<1, AGENTS>>>(t);
        cudaDeviceSynchronize();
        
        update_pheromones<<<1, 256>>>();
        cudaDeviceSynchronize();
        
        if (t % 50 == 49) {
            reset_resources<<<1, RESOURCES>>>();
            cudaDeviceSynchronize();
        }
        
        if (t % 100 == 99) {
            printf("Tick %d: Pheromones active: ", t+1);
            int pcount;
            cudaMemcpyFromSymbol(&pcount, d_pheromone_count, sizeof(int));
            printf("%d\n", pcount);
        }
    }
    
    // Gather results
    float uniform_energy, specialist_energy;
    int uniform_collected, specialist_collected;
    
    cudaMemcpyFromSymbol(&uniform_energy, d_uniform_energy, sizeof(float));
    cudaMemcpyFromSymbol(&specialist_energy, d_specialist_energy, sizeof(float));
    cudaMemcpyFromSymbol(&uniform_collected, d_uniform_collected, sizeof(int));
    cudaMemcpyFromSymbol(&specialist_collected, d_specialist_collected, sizeof(int));
    
    // Calculate fitness from agents
    float uniform_fitness = 0.0f, specialist_fitness = 0.0f;
    Agent h_agents