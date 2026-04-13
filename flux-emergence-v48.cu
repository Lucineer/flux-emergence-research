
/*
CUDA Simulation Experiment v48: Stigmergy with Pheromone Trails
Test: Agents leave pheromone markers at collected resource locations that decay over time.
Prediction: Pheromones will create positive feedback loops, allowing specialized agents to 
            outperform uniform agents by >1.61x (v8 baseline) due to collective memory.
Baseline: v8 mechanisms (scarcity, territory, comms, anti-convergence) included.
Novelty: Pheromone trails with spatial diffusion and decay.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 256; // Spatial grid for pheromone field
const float WORLD_SIZE = 1.0f;
const float DRIFT_STRENGTH = 0.01f;
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;
const float ANTI_CONVERGE_THRESH = 0.9f;
const float ENERGY_DECAY = 0.999f;

// Agent roles: [explore, collect, communicate, defend]
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];
    float fitness;
    int arch; // 0=specialized, 1=uniform
    unsigned int rng;
};

struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone grid
__device__ float pheromone[PHEROMONE_GRID][PHEROMONE_GRID];

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents and resources
__global__ void init(Agent* agents, Resource* resources, int specialized) {
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
    a->arch = (idx < AGENTS/2) ? 0 : 1; // First half specialized, second half uniform
    
    if (a->arch == 0) { // Specialized: role[arch]=0.7
        for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
        a->role[specialized] = 0.7f;
    } else { // Uniform: all roles=0.25
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
    
    // Normalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) a->role[i] /= sum;
    
    // Initialize resources (first thread only)
    if (idx == 0) {
        for (int i = 0; i < RESOURCES; i++) {
            resources[i].x = lcgf(&a->rng);
            resources[i].y = lcgf(&a->rng);
            resources[i].value = 0.8f + lcgf(&a->rng) * 0.4f;
            resources[i].collected = 0;
        }
        // Initialize pheromone grid
        for (int i = 0; i < PHEROMONE_GRID; i++) {
            for (int j = 0; j < PHEROMONE_GRID; j++) {
                pheromone[i][j] = 0.0f;
            }
        }
    }
}

// Pheromone diffusion and decay kernel
__global__ void update_pheromone() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= PHEROMONE_GRID || j >= PHEROMONE_GRID) return;
    
    // Simple diffusion and decay
    float center = pheromone[i][j];
    float decayed = center * 0.95f; // 5% decay per tick
    
    // 4-point diffusion
    float diff = 0.0f;
    int count = 0;
    if (i > 0) { diff += pheromone[i-1][j]; count++; }
    if (i < PHEROMONE_GRID-1) { diff += pheromone[i+1][j]; count++; }
    if (j > 0) { diff += pheromone[i][j-1]; count++; }
    if (j < PHEROMONE_GRID-1) { diff += pheromone[i][j+1]; count++; }
    
    if (count > 0) {
        pheromone[i][j] = decayed * 0.7f + (diff / count) * 0.3f;
    } else {
        pheromone[i][j] = decayed;
    }
    
    // Clamp
    if (pheromone[i][j] < 0.001f) pheromone[i][j] = 0.0f;
    if (pheromone[i][j] > 1.0f) pheromone[i][j] = 1.0f;
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, int tick_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // 1. Energy decay
    a->energy *= ENERGY_DECAY;
    
    // 2. Anti-convergence: check similarity with random agent
    int other_idx = (int)(lcgf(&a->rng) * AGENTS);
    if (other_idx >= AGENTS) other_idx = AGENTS - 1;
    Agent* other = &agents[other_idx];
    
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - other->role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > ANTI_CONVERGE_THRESH) {
        // Find dominant role
        int dominant = 0;
        for (int i = 1; i < 4; i++) {
            if (a->role[i] > a->role[dominant]) dominant = i;
        }
        // Apply drift to non-dominant roles
        for (int i = 0; i < 4; i++) {
            if (i != dominant) {
                float drift = (lcgf(&a->rng) - 0.5f) * DRIFT_STRENGTH;
                a->role[i] += drift;
                if (a->role[i] < 0.01f) a->role[i] = 0.01f;
            }
        }
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) a->role[i] /= sum;
    }
    
    // 3. Role-based movement
    float explore_strength = a->role[0];
    float collect_strength = a->role[1];
    float comm_strength = a->role[2];
    float defend_strength = a->role[3];
    
    // Pheromone sensing (NOVEL MECHANISM)
    int grid_x = (int)(a->x * PHEROMONE_GRID);
    int grid_y = (int)(a->y * PHEROMONE_GRID);
    if (grid_x < 0) grid_x = 0;
    if (grid_x >= PHEROMONE_GRID) grid_x = PHEROMONE_GRID - 1;
    if (grid_y < 0) grid_y = 0;
    if (grid_y >= PHEROMONE_GRID) grid_y = PHEROMONE_GRID - 1;
    
    float pheromone_val = pheromone[grid_x][grid_y];
    // Move toward pheromone if collector
    if (collect_strength > 0.3f && pheromone_val > 0.1f) {
        // Sample nearby to find gradient
        float best_val = pheromone_val;
        int best_dx = 0, best_dy = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = grid_x + dx;
                int ny = grid_y + dy;
                if (nx >= 0 && nx < PHEROMONE_GRID && ny >= 0 && ny < PHEROMONE_GRID) {
                    if (pheromone[nx][ny] > best_val) {
                        best_val = pheromone[nx][ny];
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
        }
        a->vx += best_dx * 0.005f * collect_strength;
        a->vy += best_dy * 0.005f * collect_strength;
    }
    
    // Random exploration component
    a->vx += (lcgf(&a->rng) - 0.5f) * 0.01f * explore_strength;
    a->vy += (lcgf(&a->rng) - 0.5f) * 0.01f * explore_strength;
    
    // Velocity damping and bounds
    a->vx *= 0.95f;
    a->vy *= 0.95f;
    a->x += a->vx;
    a->y += a->vy;
    
    if (a->x < 0) { a->x = 0; a->vx = fabsf(a->vx); }
    if (a->x > WORLD_SIZE) { a->x = WORLD_SIZE; a->vx = -fabsf(a->vx); }
    if (a->y < 0) { a->y = 0; a->vy = fabsf(a->vy); }
    if (a->y > WORLD_SIZE) { a->y = WORLD_SIZE; a->vy = -fabsf(a->vy); }
    
    // 4. Resource collection
    float detect_range = 0.03f + 0.04f * explore_strength;
    float grab_range = 0.02f + 0.02f * collect_strength;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range) {
            // Communicate location (v8 mechanism)
            if (comm_strength > 0.3f) {
                // Broadcast to nearby agents of same arch
                for (int j = 0; j < AGENTS; j++) {
                    if (j == idx) continue;
                    Agent* neighbor = &agents[j];
                    if (neighbor->arch == a->arch) {
                        float ndx = neighbor->x - a->x;
                        float ndy = neighbor->y - a->y;
                        if (sqrtf(ndx*ndx + ndy*ndy) < 0.06f) {
                            // Simple attraction toward resource
                            neighbor->vx += dx * 0.001f * comm_strength;
                            neighbor->vy += dy * 0.001f * comm_strength;
                        }
                    }
                }
            }
            
            if (dist < grab_range) {
                // Collect resource
                float value = r->value;
                // Collector bonus
                value *= (1.0f + 0.5f * collect_strength);
                
                // Territory defense bonus (v8)
                int defenders_nearby = 0;
                for (int j = 0; j < AGENTS; j++) {
                    if (j == idx) continue;
                    Agent* other = &agents[j];
                    if (other->arch == a->arch && other->role[3] > 0.3f) {
                        float odx = other->x - a->x;
                        float ody = other->y - a->y;
                        if (sqrtf(odx*odx + ody*ody) < 0.08f) {
                            defenders_nearby++;
                        }
                    }
                }
                value *= (1.0f + 0.2f * defenders_nearby * defend_strength);
                
                a->energy += value;
                a->fitness += value;
                r->collected = 1;
                
                // NOVEL: Leave pheromone at collected location
                int px = (int)(r->x * PHEROMONE_GRID);
                int py = (int)(r->y * PHEROMONE_GRID);
                if (px >= 0 && px < PHEROMONE_GRID && py >= 0 && py < PHEROMONE_GRID) {
                    atomicAdd(&pheromone[px][py], 0.5f);
                }
                break;
            }
        }
    }
    
    // 5. Energy-based coupling (v2/v13)
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent* other = &agents[i];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.05f) {
            float coupling = (a->arch == other->arch) ? COUPLING_SAME : COUPLING_DIFF;
            for (int j = 0; j < 4; j++) {
                float diff = other->role[j] - a->role[j];
                a->role[j] += diff * coupling * a->energy;
            }
        }
    }
    
    // 6. Periodic perturbation (every 50 ticks)
    if (tick_id % 50 == 25) {
        // Defenders resist perturbation
        float resistance = defend_strength * 0.8f;
        if (lcgf(&a->rng) > resistance) {
            a->energy *= 0.5f;
            // Small random velocity kick
            a->vx += (lcgf(&a->rng) - 0.5f) * 0.05f;
            a->vy += (lcgf(&a->rng) - 0.5f) * 0.05f;
        }
    }
    
    // 7. Resource respawn (v5)
    if (tick_id % 50 == 0 && idx < RESOURCES) {
        resources[idx].collected = 0;
        resources[idx].x = lcgf(&a->rng);
        resources[idx].y = lcgf(&a->rng);
        resources[idx].value = 0.8f + lcgf(&a->rng) * 0.4f;
    }
}

int main() {
    // Allocate memory
    Agent* agents;
    Resource* resources;
    cudaMallocManaged(&agents, AGENTS * sizeof(Agent));
    cudaMallocManaged(&resources, RESOURCES * sizeof(Resource));
    
    // Initialize
    dim3 block(256);
    dim3 grid((AGENTS + 255) / 256);
    init<<<grid, block>>>(agents, resources, 1); // Specialized as collectors
    
    cudaDeviceSynchronize();
    
    // Pheromone update blocks
    dim3 pheromone_block(16, 16);
    dim3 pheromone_grid((PHEROMONE_GRID + 15) / 16, (PHEROMONE_GRID + 15) / 16);
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        tick<<<grid, block>>>(agents, resources, t);
        update_pheromone<<<pheromone_grid, pheromone_block>>>();
        cudaDeviceSynchronize();
    }
    
    // Calculate results
    float spec_fitness = 0.0f, unif_fitness = 0.0f;
    float spec_energy = 0.0f, unif_energy = 0.0f;
    
    for (int i = 0; i < AGENTS; i++) {
        if (agents[i].arch == 0) {
            spec_fitness += agents[i].f