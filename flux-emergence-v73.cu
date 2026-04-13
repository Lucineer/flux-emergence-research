// CUDA Simulation Experiment v73: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone trails at resource locations that decay over time.
// Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents.
// Expected: Specialists should show >1.61x efficiency (v8 baseline) due to improved resource location memory.
// Novelty: Stigmergy (indirect communication through environment modification) not tested in previous experiments.

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK = 256;
const float WORLD_SIZE = 1.0f;
const float MIN_DIST = 0.0001f;

// Pheromone grid constants
const int PH_GRID = 64; // 64x64 grid
const float PH_CELL = WORLD_SIZE / PH_GRID;
const float PH_DECAY = 0.95f; // 5% decay per tick
const float PH_STRENGTH = 0.5f; // Initial strength when deposited

// Agent archetypes
enum { ARCH_GENERALIST = 0, ARCH_SPECIALIST = 1 };

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    char role[4]; // 0:explore, 1:collect, 2:communicate, 3:defend
    float fitness;
    int arch;
    unsigned int rng;
};

// Resource struct
struct Resource {
    float x, y;
    float value;
    bool collected;
};

// Pheromone grid (global memory)
__device__ float pheromone[PH_GRID * PH_GRID];

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Initialize pheromone grid
__global__ void initPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PH_GRID * PH_GRID) {
        pheromone[idx] = 0.0f;
    }
}

// Decay pheromones each tick
__global__ void decayPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PH_GRID * PH_GRID) {
        pheromone[idx] *= PH_DECAY;
    }
}

// Add pheromone at a location
__device__ void addPheromone(float x, float y) {
    int gx = (int)(x / PH_CELL) % PH_GRID;
    int gy = (int)(y / PH_CELL) % PH_GRID;
    if (gx < 0) gx += PH_GRID;
    if (gy < 0) gy += PH_GRID;
    int idx = gy * PH_GRID + gx;
    atomicAdd(&pheromone[idx], PH_STRENGTH);
}

// Sample pheromone at a location
__device__ float samplePheromone(float x, float y) {
    int gx = (int)(x / PH_CELL) % PH_GRID;
    int gy = (int)(y / PH_CELL) % PH_GRID;
    if (gx < 0) gx += PH_GRID;
    if (gy < 0) gy += PH_GRID;
    return pheromone[gy * PH_GRID + gx];
}

// Initialize agents and resources
__global__ void init(Agent *agents, Resource *resources, unsigned int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize agents
    if (idx < AGENTS) {
        Agent &a = agents[idx];
        a.rng = seed + idx * 137;
        a.x = lcgf(a.rng);
        a.y = lcgf(a.rng);
        a.vx = lcgf(a.rng) * 0.02f - 0.01f;
        a.vy = lcgf(a.rng) * 0.02f - 0.01f;
        a.energy = 1.0f;
        a.fitness = 0.0f;
        a.arch = (idx < AGENTS/2) ? ARCH_GENERALIST : ARCH_SPECIALIST;
        
        // Set roles based on archetype
        if (a.arch == ARCH_GENERALIST) {
            // Uniform roles
            a.role[0] = 0.25f;
            a.role[1] = 0.25f;
            a.role[2] = 0.25f;
            a.role[3] = 0.25f;
        } else {
            // Specialized roles (v8 baseline)
            a.role[0] = 0.7f;  // Explore
            a.role[1] = 0.1f;  // Collect
            a.role[2] = 0.1f;  // Communicate
            a.role[3] = 0.1f;  // Defend
        }
    }
    
    // Initialize resources
    if (idx < RESOURCES) {
        Resource &r = resources[idx];
        unsigned int rng = seed + idx * 7919;
        r.x = lcgf(rng);
        r.y = lcgf(rng);
        r.value = 0.5f + lcgf(rng) * 0.5f;
        r.collected = false;
    }
}

// Main simulation tick
__global__ void tick(Agent *agents, Resource *resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent &a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence (v3 mechanism)
    float role_sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    float role_norm[4];
    for (int i = 0; i < 4; i++) role_norm[i] = a.role[i] / role_sum;
    
    // Check for dominance
    int dominant = 0;
    for (int i = 1; i < 4; i++) {
        if (role_norm[i] > role_norm[dominant]) dominant = i;
    }
    
    if (role_norm[dominant] > 0.9f) {
        // Apply drift to non-dominant roles
        for (int i = 0; i < 4; i++) {
            if (i != dominant) {
                float drift = (lcgf(a.rng) - 0.5f) * 0.02f;
                a.role[i] += drift;
                if (a.role[i] < 0.01f) a.role[i] = 0.01f;
            }
        }
    }
    
    // Normalize roles
    role_sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
    for (int i = 0; i < 4; i++) a.role[i] /= role_sum;
    
    // Choose action based on role probabilities
    float action_roll = lcgf(a.rng);
    int action = 0;
    float cum_prob = a.role[0];
    
    while (action < 3 && action_roll > cum_prob) {
        action++;
        cum_prob += a.role[action];
    }
    
    // Movement with pheromone influence (NOVEL MECHANISM)
    float move_x = a.vx;
    float move_y = a.vy;
    
    // Sample pheromone in surrounding cells
    float ph_center = samplePheromone(a.x, a.y);
    float ph_right = samplePheromone(a.x + 0.02f, a.y);
    float ph_left = samplePheromone(a.x - 0.02f, a.y);
    float ph_up = samplePheromone(a.x, a.y + 0.02f);
    float ph_down = samplePheromone(a.x, a.y - 0.02f);
    
    // Move toward higher pheromone (exploit)
    if (ph_right > ph_center && ph_right > ph_left) move_x += 0.005f;
    if (ph_left > ph_center && ph_left > ph_right) move_x -= 0.005f;
    if (ph_up > ph_center && ph_up > ph_down) move_y += 0.005f;
    if (ph_down > ph_center && ph_down > ph_up) move_y -= 0.005f;
    
    // Add some randomness (explore)
    move_x += (lcgf(a.rng) - 0.5f) * 0.01f;
    move_y += (lcgf(a.rng) - 0.5f) * 0.01f;
    
    // Update position
    a.x += move_x;
    a.y += move_y;
    
    // World wrap
    if (a.x < 0) a.x += WORLD_SIZE;
    if (a.x >= WORLD_SIZE) a.x -= WORLD_SIZE;
    if (a.y < 0) a.y += WORLD_SIZE;
    if (a.y >= WORLD_SIZE) a.y -= WORLD_SIZE;
    
    // Action execution
    if (action == 0) { // Explore
        // Increased detection range for explorers
        float detect_range = 0.05f + a.role[0] * 0.02f;
        
        // Find nearest resource
        int nearest = -1;
        float nearest_dist = detect_range;
        
        for (int i = 0; i < RESOURCES; i++) {
            Resource &r = resources[i];
            if (r.collected) continue;
            
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            // Wrap distance
            if (dx > 0.5f) dx -= WORLD_SIZE;
            if (dx < -0.5f) dx += WORLD_SIZE;
            if (dy > 0.5f) dy -= WORLD_SIZE;
            if (dy < -0.5f) dy += WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest = i;
            }
        }
        
        if (nearest >= 0) {
            // Deposit pheromone at resource location (NOVEL)
            addPheromone(resources[nearest].x, resources[nearest].y);
            
            // Move toward resource
            Resource &r = resources[nearest];
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            if (dx > 0.5f) dx -= WORLD_SIZE;
            if (dx < -0.5f) dx += WORLD_SIZE;
            if (dy > 0.5f) dy -= WORLD_SIZE;
            if (dy < -0.5f) dy += WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist > MIN_DIST) {
                a.vx += dx / dist * 0.01f;
                a.vy += dy / dist * 0.01f;
            }
        }
    }
    else if (action == 1) { // Collect
        float grab_range = 0.03f + a.role[1] * 0.01f;
        
        for (int i = 0; i < RESOURCES; i++) {
            Resource &r = resources[i];
            if (r.collected) continue;
            
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            if (dx > 0.5f) dx -= WORLD_SIZE;
            if (dx < -0.5f) dx += WORLD_SIZE;
            if (dy > 0.5f) dy -= WORLD_SIZE;
            if (dy < -0.5f) dy += WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < grab_range) {
                // Collect resource
                float bonus = 1.0f + a.role[1] * 0.5f; // Up to 50% bonus
                a.energy += r.value * bonus;
                a.fitness += r.value * bonus;
                r.collected = true;
                
                // Deposit pheromone (NOVEL)
                addPheromone(r.x, r.y);
                break;
            }
        }
    }
    else if (action == 2) { // Communicate
        float comm_range = 0.06f;
        
        // Find nearest resource
        int nearest = -1;
        float nearest_dist = 1.0f;
        float rx = 0, ry = 0;
        
        for (int i = 0; i < RESOURCES; i++) {
            Resource &r = resources[i];
            if (r.collected) continue;
            
            float dx = r.x - a.x;
            float dy = r.y - a.y;
            if (dx > 0.5f) dx -= WORLD_SIZE;
            if (dx < -0.5f) dx += WORLD_SIZE;
            if (dy > 0.5f) dy -= WORLD_SIZE;
            if (dy < -0.5f) dy += WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest = i;
                rx = r.x;
                ry = r.y;
            }
        }
        
        if (nearest >= 0) {
            // Broadcast to nearby agents of same archetype
            for (int i = 0; i < AGENTS; i++) {
                if (i == idx) continue;
                Agent &other = agents[i];
                if (other.arch != a.arch) continue;
                
                float dx = other.x - a.x;
                float dy = other.y - a.y;
                if (dx > 0.5f) dx -= WORLD_SIZE;
                if (dx < -0.5f) dx += WORLD_SIZE;
                if (dy > 0.5f) dy -= WORLD_SIZE;
                if (dy < -0.5f) dy += WORLD_SIZE;
                
                float dist = sqrtf(dx*dx + dy*dy);
                if (dist < comm_range) {
                    // Influence neighbor's movement
                    float ndx = rx - other.x;
                    float ndy = ry - other.y;
                    if (ndx > 0.5f) ndx -= WORLD_SIZE;
                    if (ndx < -0.5f) ndx += WORLD_SIZE;
                    if (ndy > 0.5f) ndy -= WORLD_SIZE;
                    if (ndy < -0.5f) ndy += WORLD_SIZE;
                    
                    float ndist = sqrtf(ndx*ndx + ndy*ndy);
                    if (ndist > MIN_DIST) {
                        other.vx += ndx / ndist * 0.005f;
                        other.vy += ndy / ndist * 0.005f;
                    }
                }
            }
        }
    }
    else if (action == 3) { // Defend
        // Territory boost (v8 mechanism)
        float territory_range = 0.04f;
        int nearby_defenders = 0;
        
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent &other = agents[i];
            if (other.arch != a.arch) continue;
            
            float dx = other.x - a.x;
            float dy = other.y - a.y;
            if (dx > 0.5f) dx -= WORLD_SIZE;
            if (dx < -0.5f) dx += WORLD_SIZE;
            if (dy > 0.5f) dy -= WORLD_SIZE;
            if (dy < -0.5f) dy += WORLD_SIZE;
            
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < territory_range) {
                // Check if neighbor is also defending
                float other_action_roll = lcgf(other.rng);
                float other_cum = other.role[0];
                int other_action = 0;
                while (other_action < 3 && other_action_roll > other_cum) {
                    other_action++;
                    other_cum += other.role[other_action];
                }
                if (other_action == 3) {
                    nearby_defenders++;
                }
            }
        }
        
        // Defense bonus: 20% per nearby defender
        float defense_bonus = 1.0f + nearby_defenders * 0.2f;
        a.energy *= defense_bonus;
        
        // Perturbation resistance
        if (tick_num % 100 == 0) {
            // Generalists get full perturbation
            if (a.arch == ARCH_GENERALIST) {
                a.energy *= 0.5f;
            }
            // Specialists with defend role resist
            else if (a.role[3] < 0.3f) {
                a.energy *= 0.75f;
            }
        }
    }
    
    // Velocity damping
    a.vx *= 0.95f;
    a.vy *= 0.95f;
    
    // Keep velocity reasonable
    float speed = sqrtf(a.vx * a.v