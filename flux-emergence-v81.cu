// CUDA Simulation Experiment v81: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone markers at resource locations that decay over time.
// Prediction: Pheromones will enhance specialist coordination beyond basic communication,
//             increasing specialist advantage ratio to >1.61x (v8 baseline).
// Novelty: Stigmergy (environment-mediated indirect communication) not tested in v1-v13.

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants for sm_87 (Jetson Orin)
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Agent archetypes: 0=explorer, 1=collector, 2=communicator, 3=defender
const int ARCHETYPES = 4;

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1103515245 + 12345;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return lcg(state) / 4294967296.0f;
}

// Pheromone marker structure
struct Pheromone {
    float x, y;
    float strength;
    int arch;  // Archetype that left it
    int timer; // Decay timer
};

// Resource structure
struct Resource {
    float x, y;
    float value;
    bool collected;
};

// Agent structure
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPES];  // Behavioral tendencies
    float fitness;
    int arch;  // Dominant archetype
    unsigned int rng;
};

// Global device arrays
__device__ Agent d_agents[AGENTS];
__device__ Resource d_resources[RESOURCES];
__device__ Pheromone d_pheromones[AGENTS];  // Each agent can leave one active marker
__device__ int d_pheromone_count = 0;

// Initialize agents and resources
__global__ void init_simulation(unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < AGENTS) {
        Agent &a = d_agents[idx];
        a.rng = seed + idx * 137;
        
        // Random starting position
        a.x = lcgf(a.rng);
        a.y = lcgf(a.rng);
        a.vx = lcgf(a.rng) * 0.02f - 0.01f;
        a.vy = lcgf(a.rng) * 0.02f - 0.01f;
        a.energy = 1.0f;
        a.fitness = 0.0f;
        
        // Specialized group (first half) vs uniform control (second half)
        if (idx < AGENTS/2) {
            // Specialized: one dominant role at 0.7, others at 0.1
            a.arch = idx % ARCHETYPES;
            for (int i = 0; i < ARCHETYPES; i++) {
                a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
            }
        } else {
            // Uniform control: all roles at 0.25
            a.arch = -1;
            for (int i = 0; i < ARCHETYPES; i++) {
                a.role[i] = 0.25f;
            }
        }
    }
    
    if (idx < RESOURCES) {
        Resource &r = d_resources[idx];
        unsigned int rng_temp = d_agents[0].rng;
        r.x = lcgf(rng_temp) * 0.9f + 0.05f;
        rng_temp = d_agents[0].rng + idx * 3;
        r.y = lcgf(rng_temp) * 0.9f + 0.05f;
        rng_temp = d_agents[0].rng + idx * 7;
        r.value = 0.5f + lcgf(rng_temp) * 0.5f;
        r.collected = false;
    }
    
    if (idx == 0) {
        d_pheromone_count = 0;
    }
}

// Calculate similarity between two agents
__device__ float similarity(const Agent &a, const Agent &b) {
    float sum = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) {
        float diff = a.role[i] - b.role[i];
        sum += diff * diff;
    }
    return 1.0f / (1.0f + sqrtf(sum));
}

// Anti-convergence mechanism
__device__ void apply_anti_convergence(Agent &a, int idx) {
    // Check similarity with random other agent
    int other = (idx + 1 + (a.rng % (AGENTS-1))) % AGENTS;
    if (similarity(a, d_agents[other]) > 0.9f) {
        // Find non-dominant role with highest value
        int max_role = 0;
        float max_val = a.role[0];
        for (int i = 1; i < ARCHETYPES; i++) {
            if (a.role[i] > max_val && i != a.arch) {
                max_val = a.role[i];
                max_role = i;
            }
        }
        // Apply random drift
        float drift = lcgf(a.rng) * 0.02f - 0.01f;
        a.role[max_role] += drift;
        a.role[max_role] = fmaxf(0.05f, fminf(0.95f, a.role[max_role]));
    }
}

// Leave pheromone marker at current location
__device__ void leave_pheromone(Agent &a, int idx) {
    if (d_pheromone_count < AGENTS) {
        int slot = atomicAdd(&d_pheromone_count, 1) % AGENTS;
        Pheromone &p = d_pheromones[slot];
        p.x = a.x;
        p.y = a.y;
        p.strength = 1.0f;
        p.arch = a.arch;
        p.timer = 100;  // Lasts 100 ticks
    }
}

// Follow pheromones of same archetype
__device__ void follow_pheromones(Agent &a) {
    float best_strength = 0.0f;
    float target_x = a.x, target_y = a.y;
    
    for (int i = 0; i < min(d_pheromone_count, AGENTS); i++) {
        Pheromone &p = d_pheromones[i];
        if (p.timer > 0 && p.arch == a.arch) {
            float dx = p.x - a.x;
            float dy = p.y - a.y;
            float dist2 = dx*dx + dy*dy;
            
            if (dist2 < 0.25f && p.strength > best_strength) {
                best_strength = p.strength;
                target_x = p.x;
                target_y = p.y;
            }
        }
    }
    
    if (best_strength > 0.0f) {
        float dx = target_x - a.x;
        float dy = target_y - a.y;
        float dist = sqrtf(dx*dx + dy*dy) + 1e-6f;
        a.vx += dx / dist * 0.005f;
        a.vy += dy / dist * 0.005f;
    }
}

// Main simulation kernel
__global__ void tick(int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = d_agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Apply anti-convergence
    apply_anti_convergence(a, idx);
    
    // Update position with bounds
    a.x += a.vx;
    a.y += a.vy;
    
    if (a.x < 0.0f) { a.x = 0.0f; a.vx = fabsf(a.vx) * 0.5f; }
    if (a.x > 1.0f) { a.x = 1.0f; a.vx = -fabsf(a.vx) * 0.5f; }
    if (a.y < 0.0f) { a.y = 0.0f; a.vy = fabsf(a.vy) * 0.5f; }
    if (a.y > 1.0f) { a.y = 1.0f; a.vy = -fabsf(a.vy) * 0.5f; }
    
    // Velocity damping
    a.vx *= 0.95f;
    a.vy *= 0.95f;
    
    // Role-specific behaviors
    float explore_range = 0.03f + a.role[0] * 0.04f;
    float collect_range = 0.02f + a.role[1] * 0.02f;
    float comm_range = 0.04f + a.role[2] * 0.02f;
    
    // Explore behavior: random movement
    if (a.role[0] > 0.3f) {
        a.vx += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[0];
        a.vy += (lcgf(a.rng) - 0.5f) * 0.01f * a.role[0];
    }
    
    // STIGMERGY: Follow pheromones from same archetype
    if (tick_num > 10) {  // Let some pheromones accumulate first
        follow_pheromones(a);
    }
    
    // Resource interaction
    float best_value = 0.0f;
    int best_res = -1;
    float best_dist = 1e6f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource &r = d_resources[i];
        if (r.collected) continue;
        
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist2 = dx*dx + dy*dy;
        
        if (dist2 < explore_range * explore_range && r.value > best_value) {
            best_value = r.value;
            best_res = i;
            best_dist = dist2;
        }
    }
    
    // Move toward best resource if found
    if (best_res >= 0) {
        Resource &r = d_resources[best_res];
        float dx = r.x - a.x;
        float dy = r.y - a.y;
        float dist = sqrtf(best_dist) + 1e-6f;
        
        a.vx += dx / dist * 0.008f * a.role[0];
        a.vy += dy / dist * 0.008f * a.role[0];
        
        // Collect if in range
        if (best_dist < collect_range * collect_range) {
            float bonus = 1.0f + a.role[1] * 0.5f;  // Collector bonus
            
            // Defender territory bonus
            int defenders_nearby = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent &other = d_agents[j];
                float odx = other.x - a.x;
                float ody = other.y - a.y;
                if (odx*odx + ody*ody < 0.04f && other.arch == 3) {
                    defenders_nearby++;
                }
            }
            bonus += defenders_nearby * 0.2f;
            
            a.energy += r.value * bonus;
            a.fitness += r.value * bonus;
            r.collected = true;
            
            // STIGMERGY: Leave pheromone at resource location
            leave_pheromone(a, idx);
        }
    }
    
    // Communication behavior
    if (a.role[2] > 0.3f && best_res >= 0) {
        Resource &r = d_resources[best_res];
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent &other = d_agents[j];
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx*dx + dy*dy < comm_range * comm_range) {
                // Coupling: stronger for same archetype
                float coupling = (a.arch == other.arch) ? 0.02f : 0.002f;
                other.vx += (r.x - other.x) * 0.005f * coupling * a.role[2];
                other.vy += (r.y - other.y) * 0.005f * coupling * a.role[2];
            }
        }
    }
    
    // Defender behavior: perturbation resistance
    if (a.role[3] > 0.3f) {
        if (tick_num % 50 == idx % 50) {  // Occasional perturbation
            if (lcgf(a.rng) > a.role[3] * 0.8f) {  // Resistance based on defender role
                a.energy *= 0.5f;
                a.vx += (lcgf(a.rng) - 0.5f) * 0.1f;
                a.vy += (lcgf(a.rng) - 0.5f) * 0.1f;
            }
        }
    }
    
    // Normalize roles
    float sum = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) sum += a.role[i];
    for (int i = 0; i < ARCHETYPES; i++) a.role[i] /= sum;
}

// Update pheromones (decay and remove old ones)
__global__ void update_pheromones() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= min(d_pheromone_count, AGENTS)) return;
    
    Pheromone &p = d_pheromones[idx];
    if (p.timer > 0) {
        p.timer--;
        p.strength = p.timer / 100.0f;
    }
}

// Respawn resources periodically
__global__ void respawn_resources() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = d_resources[idx];
    if (r.collected) {
        unsigned int rng_temp = d_agents[0].rng + idx * 13;
        if (lcgf(rng_temp) < 0.05f) {  // 5% chance each tick
            rng_temp = d_agents[0].rng + idx * 13;
            r.x = lcgf(rng_temp) * 0.9f + 0.05f;
            rng_temp = d_agents[0].rng + idx * 17;
            r.y = lcgf(rng_temp) * 0.9f + 0.05f;
            rng_temp = d_agents[0].rng + idx * 19;
            r.value = 0.5f + lcgf(rng_temp) * 0.5f;
            r.collected = false;
        }
    }
}

int main() {
    printf("Experiment v81: Stigmergy with Pheromone Trails\n");
    printf("Testing: Indirect communication via environmental markers\n");
    printf("Prediction: Specialist advantage >1.61x (v8 baseline)\n");
    printf("Agents: %d (512 specialized, 512 uniform)\n", AGENTS);
    printf("Resources: %d, Ticks: %d\n\n", RESOURCES, TICKS);
    
    // No CUDA memory allocation needed (using static device arrays)
    
    // Initialize
    init_simulation<<<(AGENTS+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(123456);
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        tick<<<(AGENTS+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(t);
        update_pheromones<<<(AGENTS+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>();
        respawn_resources<<<(RESOURCES+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>();
        
        if (t % 100 == 0) {
            cudaDeviceSynchronize();
            int pheromone_count;
            cudaMemcpyFromSymbol(&pheromone_count, d_pheromone_count, sizeof(int));
            printf("Tick %d: Pheromones active: %d\n", t, pheromone_count);
        }
    }
    cudaDeviceSynchronize();
    
    // Copy results back (simplified - using single thread for reduction)
    Agent h_agents[AGENTS];
    cudaMemcpyFromSymbol(h_agents, d_agents, sizeof(Agent) * AGENTS);
    
    // Calculate statistics
    float spec_fitness = 0.0f, uniform_fitness = 0.0f;
    float spec_energy = 0.0f, uniform_energy = 0.0f;
    
    for (int i = 0; i < AGENTS; i++) {
        if