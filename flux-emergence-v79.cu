// CUDA Simulation Experiment v79: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist efficiency
// Prediction: Pheromones will help uniform agents more than specialists (falsifying advantage)
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Novel: Agents leave pheromone markers at collected resources that decay over time
//        Agents can detect pheromone intensity and follow gradients

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RES_COUNT = 128;
const int TICKS = 500;
const int PHEROMONE_GRID_SIZE = 256;
const float WORLD_SIZE = 1.0f;
const float PHEROMONE_DECAY = 0.99f;
const float PHEROMONE_STRENGTH = 0.5f;
const float PHEROMONE_DETECTION_RANGE = 0.08f;

// Agent archetypes
enum { ARCH_GENERALIST = 0, ARCH_SPECIALIST = 1 };

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return lcg(state) / 4294967296.0f;
}

// Resource structure
struct Resource {
    float x, y;
    float value;
    bool collected;
    unsigned int spawn_timer;
};

// Pheromone grid cell
struct PheromoneCell {
    float intensity[2]; // [0] for generalist, [1] for specialist
};

// Agent structure
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4]; // explore, collect, communicate, defend
    float fitness;
    int arch;
    unsigned int rng;
};

// Global device pointers
__device__ Agent* d_agents;
__device__ Resource* d_resources;
__device__ PheromoneCell* d_pheromone_grid;
__device__ int* d_specialist_collected;
__device__ int* d_generalist_collected;

// Initialize pheromone grid
__global__ void initPheromoneGrid(PheromoneCell* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE) {
        grid[idx].intensity[0] = 0.0f;
        grid[idx].intensity[1] = 0.0f;
    }
}

// Initialize agents
__global__ void initAgents(Agent* agents, int arch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    unsigned int seed = idx * 123456789 + arch * 987654321;
    
    agents[idx].x = lcgf(seed) * WORLD_SIZE;
    agents[idx].y = lcgf(seed) * WORLD_SIZE;
    agents[idx].vx = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(seed) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].arch = arch;
    agents[idx].rng = idx * 7654321 + arch * 13579;
    
    // Specialist: strong in one role (0.7), weak in others (0.1)
    // Generalist: all roles equal (0.25)
    if (arch == ARCH_SPECIALIST) {
        int specialty = idx % 4;
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = (i == specialty) ? 0.7f : 0.1f;
        }
    } else {
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void initResources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    unsigned int seed = idx * 54321 + 67890;
    resources[idx].x = lcgf(seed) * WORLD_SIZE;
    resources[idx].y = lcgf(seed) * WORLD_SIZE;
    resources[idx].value = 0.8f + lcgf(seed) * 0.4f;
    resources[idx].collected = false;
    resources[idx].spawn_timer = 0;
}

// Update pheromone grid (decay and add new)
__global__ void updatePheromoneGrid(PheromoneCell* grid, float* pheromone_additions, int arch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE) return;
    
    // Decay existing pheromones
    grid[idx].intensity[arch] *= PHEROMONE_DECAY;
    
    // Add new pheromones from this tick
    grid[idx].intensity[arch] += pheromone_additions[idx];
    
    // Clear additions for next tick
    pheromone_additions[idx] = 0.0f;
}

// Get pheromone intensity at position
__device__ float getPheromoneIntensity(float x, float y, int arch) {
    int grid_x = min(max((int)(x / WORLD_SIZE * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int grid_y = min(max((int)(y / WORLD_SIZE * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int idx = grid_y * PHEROMONE_GRID_SIZE + grid_x;
    return d_pheromone_grid[idx].intensity[arch];
}

// Add pheromone at position
__device__ void addPheromone(float x, float y, int arch, float strength) {
    int grid_x = min(max((int)(x / WORLD_SIZE * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int grid_y = min(max((int)(y / WORLD_SIZE * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int idx = grid_y * PHEROMONE_GRID_SIZE + grid_x;
    
    // Use atomic add for thread safety
    atomicAdd(&d_pheromone_grid[idx].intensity[arch], strength);
}

// Main simulation tick
__global__ void tick(int tick_num, int* specialist_collected, int* generalist_collected,
                     float* pheromone_additions_spec, float* pheromone_additions_gen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent* a = &d_agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with nearby agents
    int similar_count = 0;
    int total_nearby = 0;
    float role_sum[4] = {0};
    
    for (int i = 0; i < AGENT_COUNT; i++) {
        if (i == idx) continue;
        Agent* other = &d_agents[i];
        float dx = a->x - other->x;
        float dy = a->y - other->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.05f) {
            total_nearby++;
            float similarity = 0.0f;
            for (int r = 0; r < 4; r++) {
                similarity += fabsf(a->role[r] - other->role[r]);
                role_sum[r] += other->role[r];
            }
            similarity = 1.0f - similarity / 4.0f;
            if (similarity > 0.9f) similar_count++;
        }
    }
    
    // Apply anti-convergence drift if too similar
    if (total_nearby > 3 && similar_count > total_nearby * 0.8f) {
        // Find dominant role
        int dominant = 0;
        for (int r = 1; r < 4; r++) {
            if (a->role[r] > a->role[dominant]) dominant = r;
        }
        
        // Apply random drift to non-dominant role
        int drift_target = (dominant + 1 + (int)(lcgf(a->rng) * 3)) % 4;
        float drift = lcgf(a->rng) * 0.02f - 0.01f;
        a->role[drift_target] = max(0.05f, min(0.95f, a->role[drift_target] + drift));
        
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int r = 0; r < 4; r++) {
            a->role[r] /= sum;
        }
    }
    
    // Pheromone following behavior
    float pheromone_force_x = 0.0f;
    float pheromone_force_y = 0.0f;
    
    // Sample pheromone in surrounding points
    float angles[8] = {0.0f, 0.785f, 1.57f, 2.356f, 3.14f, 3.927f, 4.71f, 5.498f};
    float max_intensity = getPheromoneIntensity(a->x, a->y, a->arch);
    
    for (int i = 0; i < 8; i++) {
        float sample_x = a->x + cosf(angles[i]) * PHEROMONE_DETECTION_RANGE;
        float sample_y = a->y + sinf(angles[i]) * PHEROMONE_DETECTION_RANGE;
        float intensity = getPheromoneIntensity(sample_x, sample_y, a->arch);
        
        if (intensity > max_intensity * 1.1f) {
            pheromone_force_x += cosf(angles[i]) * a->role[0] * 0.5f; // Explore role affects following
            pheromone_force_y += sinf(angles[i]) * a->role[0] * 0.5f;
        }
    }
    
    // Movement with pheromone influence
    a->vx = a->vx * 0.9f + (lcgf(a->rng) * 0.02f - 0.01f) + pheromone_force_x;
    a->vy = a->vy * 0.9f + (lcgf(a->rng) * 0.02f - 0.01f) + pheromone_force_y;
    
    // Limit velocity
    float speed = sqrtf(a->vx*a->vx + a->vy*a->vy);
    if (speed > 0.02f) {
        a->vx *= 0.02f / speed;
        a->vy *= 0.02f / speed;
    }
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // World wrap
    if (a->x < 0) a->x += WORLD_SIZE;
    if (a->x >= WORLD_SIZE) a->x -= WORLD_SIZE;
    if (a->y < 0) a->y += WORLD_SIZE;
    if (a->y >= WORLD_SIZE) a->y -= WORLD_SIZE;
    
    // Resource interaction
    float best_dist = 100.0f;
    int best_res = -1;
    
    // Explore role: detection range
    float detect_range = 0.03f + a->role[0] * 0.04f;
    
    for (int r = 0; r < RES_COUNT; r++) {
        Resource* res = &d_resources[r];
        if (res->collected) continue;
        
        float dx = a->x - res->x;
        float dy = a->y - res->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_res = r;
        }
    }
    
    // Collect resource if in range
    if (best_res != -1) {
        Resource* res = &d_resources[best_res];
        float grab_range = 0.02f + a->role[1] * 0.02f;
        
        if (best_dist < grab_range) {
            // Collect resource
            float value = res->value;
            
            // Collection bonus from role
            value *= (1.0f + a->role[1] * 0.5f);
            
            // Territory bonus from nearby defenders of same archetype
            int defenders_nearby = 0;
            for (int i = 0; i < AGENT_COUNT; i++) {
                if (i == idx) continue;
                Agent* other = &d_agents[i];
                if (other->arch != a->arch) continue;
                
                float dx = a->x - other->x;
                float dy = a->y - other->y;
                float dist = sqrtf(dx*dx + dy*dy);
                
                if (dist < 0.06f && other->role[3] > 0.3f) {
                    defenders_nearby++;
                }
            }
            
            value *= (1.0f + defenders_nearby * 0.2f);
            
            a->energy += value;
            a->fitness += value;
            res->collected = true;
            res->spawn_timer = 50 + (int)(lcgf(a->rng) * 20);
            
            // Record collection for statistics
            if (a->arch == ARCH_SPECIALIST) {
                atomicAdd(specialist_collected, 1);
            } else {
                atomicAdd(generalist_collected, 1);
            }
            
            // LEAVE PHEROMONE TRAIL at collected resource location
            float* pheromone_additions = (a->arch == ARCH_SPECIALIST) ? 
                                         pheromone_additions_spec : pheromone_additions_gen;
            
            // Add to pheromone grid (will be applied in separate kernel)
            int grid_x = min(max((int)(res->x / WORLD_SIZE * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
            int grid_y = min(max((int)(res->y / WORLD_SIZE * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
            int grid_idx = grid_y * PHEROMONE_GRID_SIZE + grid_x;
            
            atomicAdd(&pheromone_additions[grid_idx], PHEROMONE_STRENGTH);
        }
    }
    
    // Communication role: broadcast resource locations
    if (a->role[2] > 0.3f && best_res != -1) {
        Resource* res = &d_resources[best_res];
        
        for (int i = 0; i < AGENT_COUNT; i++) {
            if (i == idx) continue;
            Agent* other = &d_agents[i];
            
            // Only communicate with same archetype (strong coupling)
            if (other->arch != a->arch) continue;
            
            float dx = a->x - other->x;
            float dy = a->y - other->y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < 0.06f) {
                // Influence neighbor's movement toward resource
                float influence = a->role[2] * 0.1f;
                other->vx += (res->x - other->x) * influence;
                other->vy += (res->y - other->y) * influence;
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0) {
        // Defenders resist perturbation
        float resistance = a->role[3] * 0.8f;
        if (lcgf(a->rng) > resistance) {
            a->energy *= 0.5f;
            a->vx = lcgf(a->rng) * 0.04f - 0.02f;
            a->vy = lcgf(a->rng) * 0.04f - 0.02f;
        }
    }
}

// Resource respawn
__global__ void respawnResources() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    Resource* res = &d_resources[idx];
    
    if (res->collected) {
        if (res->spawn_timer > 0) {
            res->spawn_timer--;
        } else {
            unsigned int seed = idx * 76543 + 12345;
            res->x = lcgf(seed) * WORLD_SIZE;
            res->y = lcgf(seed) * WORLD_SIZE;
            res->value = 0.8f + lcg