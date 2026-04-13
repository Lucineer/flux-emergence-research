// CUDA Simulation Experiment v45: STIGMERGY TRAILS
// Testing: Pheromone trails at resource locations to guide exploration
// Prediction: Stigmergy will amplify specialist advantage by reducing exploration costs
// Expected: Specialists with stigmergy > specialists without > uniform control
// Baseline: v8 mechanisms (scarcity, territory, comms) included

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RES_COUNT = 128;
const int TICKS = 500;
const int ARCH_COUNT = 4;
const float WORLD_SIZE = 1.0f;
const float MIN_DIST = 0.0001f;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Pheromone struct - NOVEL MECHANISM
struct Pheromone {
    float x, y;
    float strength;
    int arch;  // which archetype left it
    int age;
};

// Resource struct
struct Resource {
    float x, y;
    float value;
    bool collected;
};

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];  // explore, collect, communicate, defend
    float fitness;
    int arch;
    unsigned int rng;
};

// Global device arrays
__device__ Agent agents[AGENT_COUNT];
__device__ Resource resources[RES_COUNT];
__device__ Pheromone pheromones[RES_COUNT * 2];  // Double resource count for trails
__device__ int pheromone_count = 0;

// Initialize agents
__global__ void init_agents(bool specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    a.rng = idx * 17 + 12345;
    a.x = lcgf(a.rng) * WORLD_SIZE;
    a.y = lcgf(a.rng) * WORLD_SIZE;
    a.vx = (lcgf(a.rng) - 0.5f) * 0.02f;
    a.vy = (lcgf(a.rng) - 0.5f) * 0.02f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % ARCH_COUNT;
    
    if (specialized) {
        // Specialists: strong in their archetype's role
        for (int i = 0; i < 4; i++) a.role[i] = 0.1f;
        a.role[a.arch] = 0.7f;  // Primary role
        a.role[(a.arch + 1) % 4] = 0.15f;  // Secondary
    } else {
        // Uniform control: all roles equal
        for (int i = 0; i < 4; i++) a.role[i] = 0.25f;
    }
}

// Initialize resources
__global__ void init_resources() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    unsigned int rng = idx * 13 + 54321;
    resources[idx].x = (lcg(rng) & 0xFFFFFF) / 16777216.0f * WORLD_SIZE;
    resources[idx].y = (lcg(rng) & 0xFFFFFF) / 16777216.0f * WORLD_SIZE;
    resources[idx].value = 0.8f + lcgf(rng) * 0.4f;
    resources[idx].collected = false;
}

// Initialize pheromones (empty)
__global__ void init_pheromones() {
    pheromone_count = 0;
    for (int i = 0; i < RES_COUNT * 2; i++) {
        pheromones[i].strength = 0.0f;
        pheromones[i].age = 0;
    }
}

// Update pheromones (age and decay)
__global__ void update_pheromones() {
    for (int i = 0; i < pheromone_count; i++) {
        pheromones[i].age++;
        pheromones[i].strength *= 0.95f;  // Decay per tick
        
        if (pheromones[i].strength < 0.01f || pheromones[i].age > 100) {
            // Remove by swapping with last
            pheromones[i] = pheromones[pheromone_count - 1];
            pheromone_count--;
            i--;
        }
    }
}

// Add pheromone at location
__device__ void add_pheromone(float x, float y, int arch) {
    int idx = atomicAdd(&pheromone_count, 1);
    if (idx < RES_COUNT * 2) {
        pheromones[idx].x = x;
        pheromones[idx].y = y;
        pheromones[idx].strength = 1.0f;
        pheromones[idx].arch = arch;
        pheromones[idx].age = 0;
    }
}

// Find nearest pheromone of same archetype
__device__ bool find_pheromone(float x, float y, int arch, float &px, float &py) {
    float min_dist = 0.15f;  // Maximum detection range for pheromones
    bool found = false;
    
    for (int i = 0; i < pheromone_count; i++) {
        if (pheromones[i].arch == arch) {
            float dx = pheromones[i].x - x;
            float dy = pheromones[i].y - y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < min_dist) {
                min_dist = dist;
                px = pheromones[i].x;
                py = pheromones[i].y;
                found = true;
            }
        }
    }
    return found;
}

// Main simulation tick
__global__ void tick(bool use_stigmergy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with neighbors
    float similarity = 0.0f;
    int similar_count = 0;
    for (int i = 0; i < AGENT_COUNT; i++) {
        if (i == idx) continue;
        Agent &other = agents[i];
        float dx = other.x - a.x;
        float dy = other.y - a.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.05f) {
            float sim = 0.0f;
            for (int r = 0; r < 4; r++) {
                sim += fabsf(a.role[r] - other.role[r]);
            }
            similarity += 1.0f - sim / 4.0f;
            similar_count++;
        }
    }
    
    if (similar_count > 0) {
        similarity /= similar_count;
        if (similarity > 0.9f) {
            // Random drift on non-dominant role
            int drift_role = (a.arch + 1 + (int)(lcgf(a.rng) * 3)) % 4;
            a.role[drift_role] += (lcgf(a.rng) - 0.5f) * 0.01f;
            
            // Renormalize
            float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
            for (int r = 0; r < 4; r++) a.role[r] /= sum;
        }
    }
    
    // Movement with potential pheromone guidance
    float target_x = 0.0f, target_y = 0.0f;
    bool has_target = false;
    
    if (use_stigmergy && a.role[0] > 0.2f) {  // Explorers use pheromones
        has_target = find_pheromone(a.x, a.y, a.arch, target_x, target_y);
    }
    
    if (has_target) {
        // Move toward pheromone
        float dx = target_x - a.x;
        float dy = target_y - a.y;
        float dist = sqrtf(dx*dx + dy*dy) + MIN_DIST;
        a.vx = dx / dist * 0.01f;
        a.vy = dy / dist * 0.01f;
    } else {
        // Random walk biased by explore role
        a.vx += (lcgf(a.rng) - 0.5f) * 0.002f * a.role[0];
        a.vy += (lcgf(a.rng) - 0.5f) * 0.002f * a.role[0];
        
        // Velocity damping
        a.vx *= 0.98f;
        a.vy *= 0.98f;
    }
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World bounds
    if (a.x < 0) { a.x = 0; a.vx = fabsf(a.vx); }
    if (a.x > WORLD_SIZE) { a.x = WORLD_SIZE; a.vx = -fabsf(a.vx); }
    if (a.y < 0) { a.y = 0; a.vy = fabsf(a.vy); }
    if (a.y > WORLD_SIZE) { a.y = WORLD_SIZE; a.vy = -fabsf(a.vy); }
    
    // Resource interaction
    float best_value = 0.0f;
    int best_res = -1;
    float best_dist = 0.0f;
    
    // Detection range based on explore role
    float detect_range = 0.03f + a.role[0] * 0.04f;
    
    for (int i = 0; i < RES_COUNT; i++) {
        Resource &r = resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range) {
            float value = r.value;
            
            // Collection bonus for collectors
            if (dist < 0.02f + a.role[1] * 0.02f) {
                value *= 1.0f + a.role[1] * 0.5f;
            }
            
            // Territory bonus for defenders
            float territory_bonus = 1.0f;
            for (int j = 0; j < AGENT_COUNT; j++) {
                if (j == idx) continue;
                Agent &other = agents[j];
                if (other.arch == a.arch && other.role[3] > 0.3f) {
                    float odx = other.x - r.x;
                    float ody = other.y - r.y;
                    if (sqrtf(odx*odx + ody*ody) < 0.08f) {
                        territory_bonus += 0.2f;
                    }
                }
            }
            value *= territory_bonus;
            
            if (value > best_value) {
                best_value = value;
                best_res = i;
                best_dist = dist;
            }
        }
    }
    
    // Collect resource
    if (best_res != -1) {
        Resource &r = resources[best_res];
        
        // Grab range based on collect role
        float grab_range = 0.02f + a.role[1] * 0.02f;
        
        if (best_dist < grab_range) {
            // Collect
            float gain = r.value;
            
            // Collector bonus
            gain *= 1.0f + a.role[1] * 0.5f;
            
            a.energy += gain;
            a.fitness += gain;
            r.collected = true;
            
            // NOVEL MECHANISM: Leave pheromone at collected resource location
            if (use_stigmergy) {
                add_pheromone(r.x, r.y, a.arch);
            }
            
            // Communication to nearby agents
            for (int i = 0; i < AGENT_COUNT; i++) {
                if (i == idx) continue;
                Agent &other = agents[i];
                float dx = other.x - a.x;
                float dy = other.y - a.y;
                float dist = sqrtf(dx*dx + dy*dy);
                
                if (dist < 0.06f && a.role[2] > 0.3f) {
                    // Bias receiver toward this location
                    float influence = a.role[2] * 0.1f;
                    other.vx += (r.x - other.x) * influence;
                    other.vy += (r.y - other.y) * influence;
                }
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if ((atomicAdd(&agents[0].rng, 0) % 50) == 0) {  // Use agent0's rng as global counter
        if (lcgf(a.rng) < 0.01f) {
            // Defenders resist perturbation
            float resistance = a.role[3] * 0.8f;
            if (lcgf(a.rng) > resistance) {
                a.energy *= 0.5f;
                a.vx += (lcgf(a.rng) - 0.5f) * 0.1f;
                a.vy += (lcgf(a.rng) - 0.5f) * 0.1f;
            }
        }
    }
}

// Run experiment
void run_experiment(bool specialized, bool use_stigmergy, const char* label) {
    // Initialize
    init_agents<<<1, AGENT_COUNT>>>(specialized);
    init_resources<<<1, RES_COUNT>>>();
    init_pheromones<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Run ticks
    for (int t = 0; t < TICKS; t++) {
        tick<<<1, AGENT_COUNT>>>(use_stigmergy);
        cudaDeviceSynchronize();
        
        // Update pheromones
        if (use_stigmergy && (t % 5 == 0)) {
            update_pheromones<<<1, 1>>>();
            cudaDeviceSynchronize();
        }
        
        // Respawn resources (every 50 ticks for scarcity)
        if (t % 50 == 0 && t > 0) {
            init_resources<<<1, RES_COUNT>>>();
            cudaDeviceSynchronize();
        }
    }
    
    // Calculate statistics
    Agent host_agents[AGENT_COUNT];
    cudaMemcpyFromSymbol(host_agents, agents, sizeof(Agent) * AGENT_COUNT);
    
    float total_fitness = 0.0f;
    float avg_energy = 0.0f;
    float role_specialization = 0.0f;
    
    for (int i = 0; i < AGENT_COUNT; i++) {
        total_fitness += host_agents[i].fitness;
        avg_energy += host_agents[i].energy;
        
        // Calculate role specialization (max role strength)
        float max_role = 0.0f;
        for (int r = 0; r < 4; r++) {
            if (host_agents[i].role[r] > max_role) {
                max_role = host_agents[i].role[r];
            }
        }
        role_specialization += max_role;
    }
    
    avg_energy /= AGENT_COUNT;
    role_specialization /= AGENT_COUNT;
    
    // Get pheromone count if using stigmergy
    int phero_count = 0;
    if (use_stigmergy) {
        cudaMemcpyFromSymbol(&phero_count, pheromone_count, sizeof(int));
    }
    
    printf("%s: fitness=%.2f, energy=%.3f, spec=%.3f, pheromones=%d\n",
           label, total_fitness, avg_energy, role_specialization, phero_count);
}

int main() {
    printf("=== Experiment v45: Stigmergy Trails ===\n");
    printf("Testing: Pheromone trails at resource locations\n");
    printf("Prediction: Stigmergy amplifies specialist advantage by 20%%+\n");
    printf("Baseline: v8 mechanisms (scarcity, territory, comms) active\n\n");
    
    // Warm-up
    printf("Warming up...\n");
    run_experiment(true, false, "Warmup");
    
    // Run experiments
    printf("\nResults after %d ticks:\n", TICKS);
    
    // Control: Uniform agents without stigmergy
    run_experiment(false, false, "Uniform control (no stigmergy)");
    
    // Baseline: Specialists without stigmergy
    run_experiment(true, false, "Specialists (no stigmergy)");
    
    // Experimental: Specialists with stigmergy
    run_experiment(true, true