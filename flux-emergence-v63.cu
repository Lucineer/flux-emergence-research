/*
CUDA Simulation Experiment v63: STIGMERGY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents by >1.61x (v8 baseline) due to persistent information.
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence)
Novel: Stigmergy - agents deposit pheromone when collecting, follow pheromone gradients
Control: Specialized (role[arch]=0.7) vs Uniform (all roles=0.25)
Expected: Specialists will leverage pheromones more effectively, achieving >1.61x ratio.
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
const float ANTI_CONVERGE_THRESH = 0.9f;
const float ANTI_CONVERGE_DRIFT = 0.01f;
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;
const float ENERGY_DECAY = 0.999f;
const float PERTURB_PROB = 0.01f;

// Stigmergy constants
const int TRAIL_GRID = 64; // 64x64 grid for pheromone field
const float TRAIL_DEPOSIT = 0.5f;
const float TRAIL_DECAY = 0.95f;
const float TRAIL_SENSE_RANGE = 0.1f;
const float TRAIL_INFLUENCE = 0.3f;

// Agent roles
enum Role { EXPLORE = 0, COLLECT, COMMUNICATE, DEFEND };

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

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
    bool collected;
    unsigned int rng;  // Added RNG state for resource initialization
};

// Pheromone grid struct
struct TrailGrid {
    float trail[TRAIL_GRID][TRAIL_GRID];
};

// Kernels
__global__ void initAgents(Agent* agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    agents[idx].x = lcgf(&agents[idx].rng) * WORLD_SIZE;
    agents[idx].y = lcgf(&agents[idx].rng) * WORLD_SIZE;
    agents[idx].vx = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].arch = idx % ARCHETYPES;
    agents[idx].rng = idx * 12345 + 1;
    
    if (specialized) {
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = (i == agents[idx].arch) ? 0.7f : 0.1f;
        }
    } else {
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

__global__ void initResources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    resources[idx].rng = idx * 67890 + 1;
    resources[idx].x = lcgf(&resources[idx].rng) * WORLD_SIZE;
    resources[idx].y = lcgf(&resources[idx].rng) * WORLD_SIZE;
    resources[idx].value = 0.8f + lcgf(&resources[idx].rng) * 0.4f;
    resources[idx].collected = false;
}

__global__ void initTrails(TrailGrid* trail) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < TRAIL_GRID && y < TRAIL_GRID) {
        trail->trail[x][y] = 0.0f;
    }
}

__global__ void decayTrails(TrailGrid* trail) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < TRAIL_GRID && y < TRAIL_GRID) {
        trail->trail[x][y] *= TRAIL_DECAY;
    }
}

__device__ float senseTrail(TrailGrid* trail, float x, float y) {
    int gx = (int)((x / WORLD_SIZE) * (TRAIL_GRID - 1));
    int gy = (int)((y / WORLD_SIZE) * (TRAIL_GRID - 1));
    gx = max(0, min(TRAIL_GRID - 1, gx));
    gy = max(0, min(TRAIL_GRID - 1, gy));
    return trail->trail[gx][gy];
}

__global__ void tick(Agent* agents, Resource* resources, TrailGrid* trail, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= ENERGY_DECAY;
    
    // Anti-convergence
    float role_sum = 0.0f;
    float max_role = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < ARCHETYPES; i++) {
        role_sum += a.role[i];
        if (a.role[i] > max_role) {
            max_role = a.role[i];
            max_idx = i;
        }
    }
    float similarity = max_role / (role_sum / ARCHETYPES);
    
    if (similarity > ANTI_CONVERGE_THRESH) {
        int drift_idx;
        do {
            drift_idx = (int)(lcgf(&a.rng) * ARCHETYPES);
        } while (drift_idx == max_idx);
        
        a.role[drift_idx] += ANTI_CONVERGE_DRIFT;
        a.role[max_idx] -= ANTI_CONVERGE_DRIFT;
    }
    
    // Perturbation (defenders resist)
    if (lcgf(&a.rng) < PERTURB_PROB) {
        float resist = 1.0f - a.role[DEFEND] * 0.5f;
        a.energy *= (0.5f + 0.5f * resist);
    }
    
    // Sense pheromone gradient
    float trail_here = senseTrail(trail, a.x, a.y);
    float trail_x = 0.0f, trail_y = 0.0f;
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            float nx = a.x + dx * TRAIL_SENSE_RANGE;
            float ny = a.y + dy * TRAIL_SENSE_RANGE;
            if (nx < 0 || nx >= WORLD_SIZE || ny < 0 || ny >= WORLD_SIZE) continue;
            
            float trail_val = senseTrail(trail, nx, ny);
            if (trail_val > trail_here) {
                trail_x += dx * (trail_val - trail_here);
                trail_y += dy * (trail_val - trail_here);
            }
        }
    }
    
    // Normalize trail influence
    float trail_len = sqrtf(trail_x * trail_x + trail_y * trail_y);
    if (trail_len > 0.0f) {
        trail_x /= trail_len;
        trail_y /= trail_len;
    }
    
    // Role-based movement
    float explore_dir_x = lcgf(&a.rng) * 2.0f - 1.0f;
    float explore_dir_y = lcgf(&a.rng) * 2.0f - 1.0f;
    float explore_len = sqrtf(explore_dir_x * explore_dir_x + explore_dir_y * explore_dir_y);
    if (explore_len > 0.0f) {
        explore_dir_x /= explore_len;
        explore_dir_y /= explore_len;
    }
    
    // Combine influences
    a.vx = a.role[EXPLORE] * explore_dir_x * 0.02f +
           a.role[COLLECT] * trail_x * TRAIL_INFLUENCE * 0.02f +
           a.role[COMMUNICATE] * 0.0f +  // Communicators don't move toward trails
           a.role[DEFEND] * 0.0f;        // Defenders stay put
    
    a.vy = a.role[EXPLORE] * explore_dir_y * 0.02f +
           a.role[COLLECT] * trail_y * TRAIL_INFLUENCE * 0.02f +
           a.role[COMMUNICATE] * 0.0f +
           a.role[DEFEND] * 0.0f;
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World wrap
    if (a.x < 0) a.x = 0;
    if (a.x >= WORLD_SIZE) a.x = WORLD_SIZE - 0.001f;
    if (a.y < 0) a.y = 0;
    if (a.y >= WORLD_SIZE) a.y = WORLD_SIZE - 0.001f;
    
    // Resource collection
    float collect_range = 0.02f + a.role[COLLECT] * 0.02f;
    for (int r = 0; r < RESOURCES; r++) {
        Resource& res = resources[r];
        if (res.collected) continue;
        
        float dx = a.x - res.x;
        float dy = a.y - res.y;
        float dist = sqrtf(dx * dx + dy * dy);
        
        if (dist < collect_range) {
            float bonus = 1.0f + a.role[COLLECT] * 0.5f;
            float gain = res.value * bonus;
            a.energy += gain;
            a.fitness += gain;
            res.collected = true;
            
            // Deposit pheromone at resource location
            int gx = (int)((res.x / WORLD_SIZE) * (TRAIL_GRID - 1));
            int gy = (int)((res.y / WORLD_SIZE) * (TRAIL_GRID - 1));
            atomicAdd(&trail->trail[gx][gy], TRAIL_DEPOSIT);
            break;
        }
    }
    
    // Communication
    float comm_range = 0.06f;
    if (a.role[COMMUNICATE] > 0.3f) {
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent& other = agents[j];
            float dx = a.x - other.x;
            float dy = a.y - other.y;
            if (dx * dx + dy * dy < comm_range * comm_range) {
                // Coupling
                float coupling = (a.arch == other.arch) ? COUPLING_SAME : COUPLING_DIFF;
                for (int i = 0; i < ARCHETYPES; i++) {
                    float diff = other.role[i] - a.role[i];
                    a.role[i] += diff * coupling;
                    other.role[i] -= diff * coupling;
                }
            }
        }
    }
    
    // Territory defense bonus
    float defend_range = 0.04f;
    int nearby_defenders = 0;
    if (a.role[DEFEND] > 0.3f) {
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent& other = agents[j];
            if (other.arch != a.arch) continue;
            float dx = a.x - other.x;
            float dy = a.y - other.y;
            if (dx * dx + dy * dy < defend_range * defend_range) {
                if (other.role[DEFEND] > 0.3f) {
                    nearby_defenders++;
                }
            }
        }
        float defense_bonus = 1.0f + nearby_defenders * 0.2f;
        a.energy *= defense_bonus;
        a.fitness *= defense_bonus;
    }
}

int main() {
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    TrailGrid* d_trail_spec;
    TrailGrid* d_trail_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_trail_spec, sizeof(TrailGrid));
    cudaMalloc(&d_trail_uniform, sizeof(TrailGrid));
    
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    
    // Initialize
    dim3 block(256);
    dim3 grid((AGENTS + 255) / 256);
    
    initAgents<<<grid, block>>>(d_agents_spec, 1);
    initAgents<<<grid, block>>>(d_agents_uniform, 0);
    initResources<<<grid, block>>>(d_resources);
    
    dim3 trailBlock(16, 16);
    dim3 trailGrid((TRAIL_GRID + 15) / 16, (TRAIL_GRID + 15) / 16);
    initTrails<<<trailGrid, trailBlock>>>(d_trail_spec);
    initTrails<<<trailGrid, trailBlock>>>(d_trail_uniform);
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        // Specialized group
        decayTrails<<<trailGrid, trailBlock>>>(d_trail_spec);
        tick<<<grid, block>>>(d_agents_spec, d_resources, d_trail_spec, t);
        
        // Uniform group
        decayTrails<<<trailGrid, trailBlock>>>(d_trail_uniform);
        tick<<<grid, block>>>(d_agents_uniform, d_resources + RESOURCES/2, d_trail_uniform, t);
        
        // Reset resources every 50 ticks (v5 mechanism)
        if (t % 50 == 49) {
            initResources<<<grid, block>>>(d_resources);
            initResources<<<grid, block>>>(d_resources + RESOURCES/2);
        }
        
        cudaDeviceSynchronize();
    }
    
    // Copy back results
    cudaMemcpy(h_agents_spec, d_agents_spec, AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_agents_uniform, d_agents_uniform, AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float spec_fitness = 0.0f;
    float uniform_fitness = 0.0f;
    float spec_energy = 0.0f;
    float uniform_energy = 0.0f;
    
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
    
    // Calculate specialization metric
    float spec_specialization = 0.0f;
    for (int i = 0; i < AGENTS; i++) {
        float max_role = 0.0f;
        float sum_role = 0.0f;
        for (int r = 0; r < ARCHETYPES; r++) {
            sum_role += h_agents_spec[i].role[r];
            if (h_agents_spec[i].role[r] > max_role) max_role = h_agents_spec[i].role[r];
        }
        spec_specialization += max_role / (sum_role / ARCHETYPES);
    }
    spec_specialization /= AGENTS