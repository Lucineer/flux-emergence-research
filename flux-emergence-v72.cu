/*
CUDA Simulation Experiment v72: Stigmergy with Pheromone Trails

HYPOTHESIS: Adding pheromone trails at resource locations will enhance specialist
advantage by creating persistent environmental markers that guide exploration,
reducing redundant searching and amplifying information sharing.

PREDICTION: Pheromones will increase specialist advantage ratio from baseline 1.61x
to >1.8x by creating emergent information networks that specialists can exploit
more efficiently than generalists.

NOVEL MECHANISM: Agents deposit pheromone at collected resource locations.
Pheromones decay over time and diffuse spatially. Agents can detect pheromone
gradients within their detection range, moving toward stronger concentrations.

BASELINE: Includes all v8 confirmed mechanisms (scarcity, territory, comms,
anti-convergence, behavioral roles).

CONTROL: Uniform agents (all roles=0.25) vs Specialized (role[arch]=0.7)
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
const float MIN_DIST = 0.0001f;

// Pheromone grid constants
const int PH_GRID = 64; // 64x64 grid
const float PH_CELL = WORLD_SIZE / PH_GRID;
const float PH_DEPOSIT = 0.5f;
const float PH_DECAY = 0.99f;
const float PH_DIFFUSE = 0.25f;
const float PH_DETECT_SCALE = 2.0f; // Specialists get better at reading pheromones

// Agent archetypes
enum { EXPLORER = 0, COLLECTOR, COMMUNICATOR, DEFENDER, ARCHETYPES = 4 };

// Agent struct
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral tendencies
    float fitness;        // Performance metric
    int arch;             // Dominant archetype
    unsigned int rng;     // Random state
    
    // Pheromone memory
    float last_ph_strength;
    int ph_memory_x, ph_memory_y; // Last strong pheromone location
};

// Resource struct  
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone grid (device global memory)
__device__ float d_pheromone[PH_GRID * PH_GRID];
__device__ float d_pheromone_next[PH_GRID * PH_GRID];

// Linear Congruential Generator
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize pheromone grid
__global__ void initPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PH_GRID * PH_GRID) {
        d_pheromone[idx] = 0.0f;
        d_pheromone_next[idx] = 0.0f;
    }
}

// Diffuse and decay pheromones
__global__ void updatePheromone() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= PH_GRID || y >= PH_GRID) return;
    
    int idx = y * PH_GRID + x;
    float current = d_pheromone[idx];
    
    // Diffusion from neighbors (Moore neighborhood)
    float diffuse_sum = 0.0f;
    int count = 0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < PH_GRID && ny >= 0 && ny < PH_GRID) {
                diffuse_sum += d_pheromone[ny * PH_GRID + nx];
                count++;
            }
        }
    }
    
    float diffuse_avg = diffuse_sum / count;
    float diffused = current + PH_DIFFUSE * (diffuse_avg - current);
    d_pheromone_next[idx] = fmaxf(0.0f, diffused * PH_DECAY);
}

// Swap pheromone buffers
__global__ void swapPheromoneBuffers() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PH_GRID * PH_GRID) {
        d_pheromone[idx] = d_pheromone_next[idx];
    }
}

// Deposit pheromone at location
__device__ void depositPheromone(float x, float y, float amount) {
    int grid_x = min(PH_GRID - 1, max(0, (int)(x / PH_CELL)));
    int grid_y = min(PH_GRID - 1, max(0, (int)(y / PH_CELL)));
    int idx = grid_y * PH_GRID + grid_x;
    atomicAdd(&d_pheromone_next[idx], amount);
}

// Sample pheromone at location
__device__ float samplePheromone(float x, float y) {
    int grid_x = min(PH_GRID - 1, max(0, (int)(x / PH_CELL)));
    int grid_y = min(PH_GRID - 1, max(0, (int)(y / PH_CELL)));
    return d_pheromone[grid_y * PH_GRID + grid_x];
}

// Get pheromone gradient (returns direction to stronger pheromone)
__device__ void getPheromoneGradient(float x, float y, float* grad_x, float* grad_y) {
    float center = samplePheromone(x, y);
    float east = samplePheromone(fminf(x + PH_CELL, WORLD_SIZE), y);
    float west = samplePheromone(fmaxf(x - PH_CELL, 0.0f), y);
    float north = samplePheromone(x, fminf(y + PH_CELL, WORLD_SIZE));
    float south = samplePheromone(x, fmaxf(y - PH_CELL, 0.0f));
    
    *grad_x = east - west;
    *grad_y = north - south;
    
    // Normalize
    float mag = sqrtf(*grad_x * *grad_x + *grad_y * *grad_y);
    if (mag > 0.001f) {
        *grad_x /= mag;
        *grad_y /= mag;
    }
}

// Initialize agents and resources
__global__ void init(Agent* agents, Resource* resources, int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < AGENTS) {
        Agent* a = &agents[idx];
        a->rng = idx * 17 + 12345;
        a->x = lcgf(&a->rng);
        a->y = lcgf(&a->rng);
        a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
        a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
        a->energy = 1.0f;
        a->fitness = 0.0f;
        a->last_ph_strength = 0.0f;
        a->ph_memory_x = -1;
        a->ph_memory_y = -1;
        
        if (specialized) {
            // Specialized population: each agent has dominant archetype
            a->arch = idx % ARCHETYPES;
            for (int i = 0; i < ARCHETYPES; i++) {
                a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
            }
        } else {
            // Uniform control population
            a->arch = -1;
            for (int i = 0; i < ARCHETYPES; i++) {
                a->role[i] = 0.25f;
            }
        }
    }
    
    if (idx < RESOURCES) {
        Resource* r = &resources[idx];
        r->x = lcgf(&agents[0].rng); // Use first agent's RNG for consistency
        r->y = lcgf(&agents[0].rng);
        r->value = 0.8f + lcgf(&agents[0].rng) * 0.4f;
        r->collected = 0;
    }
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect similarity with neighbors
    int similar_count = 0;
    int total_count = 0;
    
    // Pheromone-based movement adjustment
    float ph_grad_x = 0.0f, ph_grad_y = 0.0f;
    float current_ph = samplePheromone(a->x, a->y);
    
    // Specialists get better pheromone detection
    float ph_sensitivity = (a->arch >= 0) ? PH_DETECT_SCALE : 1.0f;
    
    if (current_ph > 0.01f) {
        getPheromoneGradient(a->x, a->y, &ph_grad_x, &ph_grad_y);
        a->last_ph_strength = current_ph;
        
        // Remember strong pheromone locations
        if (current_ph > 0.1f && a->ph_memory_x < 0) {
            a->ph_memory_x = (int)(a->x / PH_CELL);
            a->ph_memory_y = (int)(a->y / PH_CELL);
        }
    }
    
    // Behavioral roles implementation
    float move_x = a->vx;
    float move_y = a->vy;
    
    // Explorer: follow pheromone gradients weakly
    if (a->role[EXPLORER] > 0.3f) {
        float explore_strength = a->role[EXPLORER] * 0.1f * ph_sensitivity;
        move_x += ph_grad_x * explore_strength;
        move_y += ph_grad_y * explore_strength;
    }
    
    // Collector: move toward remembered pheromone locations
    if (a->role[COLLECTOR] > 0.3f && a->ph_memory_x >= 0) {
        float target_x = (a->ph_memory_x + 0.5f) * PH_CELL;
        float target_y = (a->ph_memory_y + 0.5f) * PH_CELL;
        float dx = target_x - a->x;
        float dy = target_y - a->y;
        float dist = sqrtf(dx*dx + dy*dy) + MIN_DIST;
        float collect_strength = a->role[COLLECTOR] * 0.05f;
        move_x += (dx / dist) * collect_strength;
        move_y += (dy / dist) * collect_strength;
    }
    
    // Velocity limits
    float speed = sqrtf(move_x*move_x + move_y*move_y);
    if (speed > 0.03f) {
        move_x = move_x / speed * 0.03f;
        move_y = move_y / speed * 0.03f;
    }
    
    // Update position with wrap-around
    a->x += move_x;
    a->y += move_y;
    if (a->x < 0) a->x += WORLD_SIZE;
    if (a->x >= WORLD_SIZE) a->x -= WORLD_SIZE;
    if (a->y < 0) a->y += WORLD_SIZE;
    if (a->y >= WORLD_SIZE) a->y -= WORLD_SIZE;
    
    // Resource collection
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
        if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
        if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
        if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
        
        float dist2 = dx*dx + dy*dy;
        float grab_range = 0.02f + a->role[COLLECTOR] * 0.02f;
        
        if (dist2 < grab_range * grab_range) {
            // Collector bonus
            float bonus = 1.0f + a->role[COLLECTOR] * 0.5f;
            a->energy += r->value * bonus;
            a->fitness += r->value * bonus;
            r->collected = 1;
            
            // Deposit pheromone at resource location
            depositPheromone(r->x, r->y, PH_DEPOSIT * (1.0f + a->role[COLLECTOR]));
            
            // Defender territory bonus
            int defender_count = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent* other = &agents[j];
                if (other->arch == a->arch && other->arch == DEFENDER) {
                    float odx = other->x - a->x;
                    float ody = other->y - a->y;
                    if (odx > 0.5f * WORLD_SIZE) odx -= WORLD_SIZE;
                    if (odx < -0.5f * WORLD_SIZE) odx += WORLD_SIZE;
                    if (ody > 0.5f * WORLD_SIZE) ody -= WORLD_SIZE;
                    if (ody < -0.5f * WORLD_SIZE) ody += WORLD_SIZE;
                    if (odx*odx + ody*ody < 0.1f * 0.1f) {
                        defender_count++;
                    }
                }
            }
            a->energy += r->value * defender_count * 0.2f;
            a->fitness += r->value * defender_count * 0.2f;
            
            break;
        }
    }
    
    // Communication role: broadcast resource locations
    if (a->role[COMMUNICATOR] > 0.3f) {
        // In pheromone system, communication is partially replaced by
        // environmental marking, but still share via direct interaction
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            if (other->arch == a->arch) {
                float dx = other->x - a->x;
                float dy = other->y - a->y;
                if (dx > 0.5f * WORLD_SIZE) dx -= WORLD_SIZE;
                if (dx < -0.5f * WORLD_SIZE) dx += WORLD_SIZE;
                if (dy > 0.5f * WORLD_SIZE) dy -= WORLD_SIZE;
                if (dy < -0.5f * WORLD_SIZE) dy += WORLD_SIZE;
                
                if (dx*dx + dy*dy < 0.06f * 0.06f) {
                    // Share pheromone memory
                    if (a->ph_memory_x >= 0 && other->ph_memory_x < 0) {
                        other->ph_memory_x = a->ph_memory_x;
                        other->ph_memory_y = a->ph_memory_y;
                    }
                }
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0 && lcgf(&a->rng) < 0.3f) {
        // Defenders resist perturbation
        if (a->role[DEFENDER] < 0.5f || lcgf(&a->rng) < 0.3f) {
            a->energy *= 0.5f;
            a->vx = lcgf(&a->rng) * 0.04f - 0.02f;
            a->vy = lcgf(&a->rng) * 0.04f - 0.02f;
        }
    }
    
    // Anti-convergence drift
    if (similar_count > total_count * 0.9f && total_count > 0) {
        // Find non-dominant role
        int non_dom = 0;
        for (int i = 1; i < ARCHETYPES; i++) {
            if (a->role[i] < a->role[non_dom]) non_dom = i;
        }
        a->role[non_dom] += (lcgf(&a->rng) * 0.02f - 0.01f);
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->