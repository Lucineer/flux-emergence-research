/*
CUDA Simulation Experiment v84: STIGMERY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents by >1.61x (v8 baseline) due to indirect communication.
Baseline: v8 mechanisms (scarcity, territory, comms, anti-convergence) included.
Novelty: Stigmergy - agents deposit pheromones when collecting resources, 
         other agents sense pheromone gradients to find resources.
Control: Specialized agents (role[arch]=0.7) vs uniform (all roles=0.25).
Expected: Specialists should leverage pheromones better due to role differentiation.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const int PHEROMONE_GRID = 64; // 64x64 grid for pheromone map
const float WORLD_SIZE = 1.0f;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

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
    float role[ARCHETYPES];
    float fitness;
    int arch;
    unsigned int rng;
    
    __device__ void apply_force(float fx, float fy, float max_speed = 0.02f) {
        vx += fx;
        vy += fy;
        float speed = sqrtf(vx*vx + vy*vy);
        if (speed > max_speed) {
            vx = vx * max_speed / speed;
            vy = vy * max_speed / speed;
        }
    }
};

// Pheromone grid (global memory)
__device__ float pheromone[PHEROMONE_GRID][PHEROMONE_GRID];

// Kernels
__global__ void init_agents(Agent *agents, bool specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    a.rng = idx * 17 + 12345;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = (lcgf(a.rng)-0.5f)*0.01f;
    a.vy = (lcgf(a.rng)-0.5f)*0.01f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % ARCHETYPES;
    
    if (specialized) {
        for (int i = 0; i < ARCHETYPES; i++) {
            a.role[i] = (i == a.arch) ? 0.7f : 0.1f;
        }
    } else {
        for (int i = 0; i < ARCHETYPES; i++) {
            a.role[i] = 0.25f;
        }
    }
}

__global__ void init_resources(Resource *resources, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = resources[idx];
    unsigned int rng = idx * 13 + 54321 + seed;
    r.x = lcgf(rng);
    r.y = lcgf(rng);
    r.value = 0.8f + lcgf(rng)*0.4f;
    r.collected = false;
}

__global__ void clear_pheromones() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < PHEROMONE_GRID && y < PHEROMONE_GRID) {
        pheromone[x][y] *= 0.95f; // Decay
    }
}

__global__ void tick(Agent *agents, Resource *resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: check similarity with random agent
    int other_idx = (lcg(a.rng) % AGENTS);
    Agent &other = agents[other_idx];
    float similarity = 0.0f;
    for (int i = 0; i < ARCHETYPES; i++) {
        similarity += fabsf(a.role[i] - other.role[i]);
    }
    similarity = 1.0f - similarity / ARCHETYPES;
    
    if (similarity > 0.9f) {
        int drift_role = lcg(a.rng) % ARCHETYPES;
        while (drift_role == a.arch) drift_role = lcg(a.rng) % ARCHETYPES;
        a.role[drift_role] += (lcgf(a.rng)-0.5f)*0.01f;
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < ARCHETYPES; i++) sum += a.role[i];
        for (int i = 0; i < ARCHETYPES; i++) a.role[i] /= sum;
    }
    
    // Movement forces
    float fx = 0.0f, fy = 0.0f;
    
    // Explore role (arch 0)
    if (a.role[0] > 0.2f) {
        float explore_range = 0.03f + a.role[0]*0.04f;
        float closest_dist = 1e6;
        int closest_res = -1;
        
        for (int i = 0; i < RESOURCES; i++) {
            Resource &r = resources[i];
            if (r.collected) continue;
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            dx -= floorf(dx + 0.5f); // Toroidal wrap
            dy -= floorf(dy + 0.5f);
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < explore_range && dist < closest_dist) {
                closest_dist = dist;
                closest_res = i;
            }
        }
        
        if (closest_res >= 0) {
            Resource &r = resources[closest_res];
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            dx -= floorf(dx + 0.5f);
            dy -= floorf(dy + 0.5f);
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist > 0.001f) {
                fx += dx/dist * a.role[0];
                fy += dy/dist * a.role[0];
            }
        }
    }
    
    // Collect role (arch 1) - also deposits pheromones
    if (a.role[1] > 0.2f) {
        float grab_range = 0.02f + a.role[1]*0.02f;
        for (int i = 0; i < RESOURCES; i++) {
            Resource &r = resources[i];
            if (r.collected) continue;
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            dx -= floorf(dx + 0.5f);
            dy -= floorf(dy + 0.5f);
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < grab_range) {
                // Collect
                a.energy += r.value * (1.0f + 0.5f * a.role[1]); // 50% bonus
                a.fitness += r.value;
                r.collected = true;
                
                // STIGMERY: Deposit pheromone at collection site
                int px = (int)((r.x + 0.5f) * PHEROMONE_GRID) % PHEROMONE_GRID;
                int py = (int)((r.y + 0.5f) * PHEROMONE_GRID) % PHEROMONE_GRID;
                atomicAdd(&pheromone[px][py], 0.5f + a.role[1]);
            }
        }
    }
    
    // Communicate role (arch 2)
    if (a.role[2] > 0.2f) {
        float comm_range = 0.06f;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            dx -= floorf(dx + 0.5f);
            dy -= floorf(dy + 0.5f);
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < comm_range) {
                // Coupling
                float coupling = (a.arch == other.arch) ? 0.02f : 0.002f;
                for (int j = 0; j < ARCHETYPES; j++) {
                    a.role[j] += (other.role[j] - a.role[j]) * coupling;
                }
                // Normalize
                float sum = 0.0f;
                for (int j = 0; j < ARCHETYPES; j++) sum += a.role[j];
                for (int j = 0; j < ARCHETYPES; j++) a.role[j] /= sum;
            }
        }
    }
    
    // Defend role (arch 3) - also senses pheromones
    if (a.role[3] > 0.2f) {
        // Territory bonus
        float defend_range = 0.04f;
        int nearby_defenders = 0;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            if (other.arch != a.arch) continue;
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            dx -= floorf(dx + 0.5f);
            dy -= floorf(dy + 0.5f);
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < defend_range && other.role[3] > 0.2f) {
                nearby_defenders++;
            }
        }
        float territory_bonus = 1.0f + nearby_defenders * 0.2f;
        a.energy *= territory_bonus;
        
        // STIGMERY: Sense pheromone gradient
        int px = (int)((a.x + 0.5f) * PHEROMONE_GRID) % PHEROMONE_GRID;
        int py = (int)((a.y + 0.5f) * PHEROMONE_GRID) % PHEROMONE_GRID;
        
        float center = pheromone[px][py];
        float east = pheromone[(px+1)%PHEROMONE_GRID][py];
        float west = pheromone[(px-1+PHEROMONE_GRID)%PHEROMONE_GRID][py];
        float north = pheromone[px][(py+1)%PHEROMONE_GRID];
        float south = pheromone[px][(py-1+PHEROMONE_GRID)%PHEROMONE_GRID];
        
        fx += (east - west) * 0.1f * a.role[3];
        fy += (north - south) * 0.1f * a.role[3];
    }
    
    // Apply movement
    a.apply_force(fx, fy);
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    a.x -= floorf(a.x);
    a.y -= floorf(a.y);
    
    // Perturbation every 50 ticks
    if (tick_num % 50 == 0) {
        float resistance = a.role[3] > 0.2f ? 0.7f : 1.0f;
        a.energy *= (0.5f * resistance + 0.5f);
        a.vx += (lcgf(a.rng)-0.5f)*0.02f;
        a.vy += (lcgf(a.rng)-0.5f)*0.02f;
    }
    
    // Resource respawn
    if (tick_num % 50 == 0) {
        for (int i = 0; i < RESOURCES; i++) {
            if (resources[i].collected) {
                resources[i].x = lcgf(a.rng);
                resources[i].y = lcgf(a.rng);
                resources[i].value = 0.8f + lcgf(a.rng)*0.4f;
                resources[i].collected = false;
            }
        }
    }
}

int main() {
    // Allocate
    Agent *d_agents_spec, *d_agents_uniform;
    Resource *d_resources_spec, *d_resources_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources_spec, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_resources_uniform, RESOURCES * sizeof(Resource));
    
    // Initialize
    init_agents<<<(AGENTS+255)/256, 256>>>(d_agents_spec, true);
    init_agents<<<(AGENTS+255)/256, 256>>>(d_agents_uniform, false);
    init_resources<<<(RESOURCES+255)/256, 256>>>(d_resources_spec, 0);
    init_resources<<<(RESOURCES+255)/256, 256>>>(d_resources_uniform, 1);
    
    // Clear pheromones
    dim3 grid((PHEROMONE_GRID+15)/16, (PHEROMONE_GRID+15)/16);
    dim3 block(16, 16);
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        clear_pheromones<<<grid, block>>>();
        tick<<<(AGENTS+255)/256, 256>>>(d_agents_spec, d_resources_spec, t);
        tick<<<(AGENTS+255)/256, 256>>>(d_agents_uniform, d_resources_uniform, t);
        cudaDeviceSynchronize();
    }
    
    // Retrieve results
    Agent *h_agents_spec = new Agent[AGENTS];
    Agent *h_agents_uniform = new Agent[AGENTS];
    cudaMemcpy(h_agents_spec, d_agents_spec, AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_agents_uniform, d_agents_uniform, AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float spec_fitness = 0.0f, uniform_fitness = 0.0f;
    float spec_energy = 0.0f, uniform_energy = 0.0f;
    
    for (int i = 0; i < AGENTS; i++) {
        spec_fitness += h_agents_spec[i].fitness;
        uniform_fitness += h_agents_uniform[i].fitness;
        spec_energy += h_agents_spec[i].energy;
        uniform_energy += h_agents_uniform[i].energy;
    }
    
    spec_fitness /= AGENTS;
    uniform_fitness /= AGENTS;
    spec_energy /= AGENTS;
    uniform_energy /= AGENTS;
    
    // Print results
    printf("\n=== EXPERIMENT v84: STIGMERY TRAILS ===\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n", AGENTS, RESOURCES, TICKS);
    printf("Pheromone grid: %dx%d, Decay: 0.95/tick\n", PHEROMONE_GRID, PHEROMONE_GRID);
    printf("Specialized agents: role[arch]=0.7, others=0.1\n");
    printf("Uniform agents: all roles=0.25\n");
    printf("Novel mechanism: Agents deposit pheromones at collection sites,\n");
    printf("                 defenders sense gradient for navigation.\n\n");
    
    printf("RESULTS:\n");
    printf("Specialized avg fitness: %.3f\n", spec_fitness);
    printf("Uniform avg fitness:     %.3f\n", uniform_fitness);
    printf("Advantage ratio:         %.3fx\n", spec_fitness / uniform_fitness);
    printf("Specialized avg energy:  %.3f\n", spec_energy);
    printf("Uniform avg energy:      %.3f\n", uniform_energy);
    
    printf("\nINTERPRETATION:\n");
    float ratio = spec_fitness / uniform_fitness;
    if (ratio > 1.61f) {
        printf("CONFIR