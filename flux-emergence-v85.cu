
/*
CUDA Simulation Experiment v85: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at collected resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents.
Baseline: v8 mechanisms (scarcity, territory, communication) included.
Novelty: Pheromone trails that agents can detect and follow.
Comparison: Specialized archetypes (role[arch]=0.7) vs uniform control (all roles=0.25).
Expected: Specialists will use pheromones more effectively, increasing fitness ratio >1.61x.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const int BLOCK_SIZE = 256;

// Pheromone grid constants
const int PH_GRID_SIZE = 64; // 64x64 grid
const float WORLD_SIZE = 1.0f;
const float PH_CELL_SIZE = WORLD_SIZE / PH_GRID_SIZE;
const float PH_DECAY = 0.95f; // per tick
const float PH_DEPOSIT = 1.0f;
const float PH_DETECTION_RANGE = 0.08f;

// Agent struct
struct Agent {
    float x, y;           // position
    float vx, vy;         // velocity
    float energy;         // energy level
    float role[4];        // behavioral roles: explore, collect, communicate, defend
    float fitness;        // accumulated resource value
    int arch;             // archetype 0-3
    unsigned int rng;     // random state
};

// Resource struct
struct Resource {
    float x, y;           // position
    float value;          // resource value
    int collected;        // 0=available, 1=collected
};

// Pheromone struct for grid cell
struct Pheromone {
    float strength[ARCHETYPES]; // pheromone strength per archetype
};

// Linear congruential generator (device/host)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return (lcg(state) & 0xFFFF) / 65535.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    a->rng = idx * 17 + 12345;
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->vy = (lcgf(&a->rng) - 0.5f) * 0.02f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    a->arch = idx % ARCHETYPES;
    
    // Specialized agents (first half): role[arch]=0.7, others 0.1
    // Uniform agents (second half): all roles=0.25
    if (idx < AGENTS/2) {
        for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
        a->role[a->arch] = 0.7f;
    } else {
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    unsigned int seed = idx * 19 + 54321;
    r->x = (seed * 1103515245 + 12345) & 0xFFFF;
    r->x = r->x / 65535.0f;
    r->y = ((seed+1) * 1103515245 + 12345) & 0xFFFF;
    r->y = r->y / 65535.0f;
    r->value = 0.8f + 0.4f * (seed % 100) / 100.0f;
    r->collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PH_GRID_SIZE * PH_GRID_SIZE) return;
    
    for (int i = 0; i < ARCHETYPES; i++) {
        pheromones[idx].strength[i] = 0.0f;
    }
}

// Decay pheromones kernel
__global__ void decay_pheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PH_GRID_SIZE * PH_GRID_SIZE) return;
    
    for (int i = 0; i < ARCHETYPES; i++) {
        pheromones[idx].strength[i] *= PH_DECAY;
    }
}

// Get pheromone grid index
__device__ int get_ph_index(float x, float y) {
    int xi = (int)(x / PH_CELL_SIZE) % PH_GRID_SIZE;
    int yi = (int)(y / PH_CELL_SIZE) % PH_GRID_SIZE;
    return yi * PH_GRID_SIZE + xi;
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromones, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect similarity > 0.9 and drift non-dominant roles
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += (a->role[i] > 0.6f) ? 1.0f : 0.0f;
    }
    similarity /= 4.0f;
    
    if (similarity > 0.9f) {
        for (int i = 0; i < 4; i++) {
            if (a->role[i] < 0.5f) {
                a->role[i] += (lcgf(&a->rng) - 0.5f) * 0.01f;
                a->role[i] = fmaxf(0.0f, fminf(1.0f, a->role[i]));
            }
        }
    }
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick_num % 50 == 0) {
        if (a->role[3] < 0.3f) { // not a defender
            a->energy *= 0.5f;
        }
    }
    
    // Movement with pheromone influence
    float ph_influence_x = 0.0f;
    float ph_influence_y = 0.0f;
    
    // Sample pheromones in detection range
    int samples = 5;
    for (int s = 0; s < samples; s++) {
        float angle = 2.0f * M_PI * lcgf(&a->rng);
        float dist = PH_DETECTION_RANGE * lcgf(&a->rng);
        float sx = a->x + cosf(angle) * dist;
        float sy = a->y + sinf(angle) * dist;
        
        if (sx < 0 || sx >= WORLD_SIZE || sy < 0 || sy >= WORLD_SIZE) continue;
        
        int ph_idx = get_ph_index(sx, sy);
        float ph_strength = pheromones[ph_idx].strength[a->arch];
        
        ph_influence_x += (sx - a->x) * ph_strength;
        ph_influence_y += (sy - a->y) * ph_strength;
    }
    
    // Normalize and apply pheromone influence (weighted by explore role)
    float ph_mag = sqrtf(ph_influence_x*ph_influence_x + ph_influence_y*ph_influence_y);
    if (ph_mag > 0.001f) {
        ph_influence_x = ph_influence_x / ph_mag * a->role[0] * 0.02f;
        ph_influence_y = ph_influence_y / ph_mag * a->role[0] * 0.02f;
    }
    
    // Update velocity with pheromone influence and random exploration
    a->vx = a->vx * 0.9f + (lcgf(&a->rng)-0.5f)*0.01f + ph_influence_x;
    a->vy = a->vy * 0.9f + (lcgf(&a->rng)-0.5f)*0.01f + ph_influence_y;
    
    // Limit speed
    float speed = sqrtf(a->vx*a->vx + a->vy*a->vy);
    if (speed > 0.03f) {
        a->vx *= 0.03f / speed;
        a->vy *= 0.03f / speed;
    }
    
    // Update position (toroidal world)
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0) a->x += WORLD_SIZE;
    if (a->x >= WORLD_SIZE) a->x -= WORLD_SIZE;
    if (a->y < 0) a->y += WORLD_SIZE;
    if (a->y >= WORLD_SIZE) a->y -= WORLD_SIZE;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    // Explore role detection range: 0.03-0.07
    float detect_range = 0.03f + a->role[0] * 0.04f;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Toroidal distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
    }
    
    if (best_res != -1 && best_dist < detect_range) {
        Resource* r = &resources[best_res];
        
        // Collect role grab range: 0.02-0.04
        float grab_range = 0.02f + a->role[1] * 0.02f;
        
        if (best_dist < grab_range && !r->collected) {
            // Collect resource
            float value = r->value;
            // Collect bonus: +50% for high collect role
            value *= (1.0f + 0.5f * a->role[1]);
            
            // Territory bonus: +20% per nearby same-arch defender
            int nearby_defenders = 0;
            for (int j = 0; j < AGENTS; j += AGENTS/16) { // sample
                if (j == idx) continue;
                Agent* other = &agents[j];
                if (other->arch == a->arch && other->role[3] > 0.5f) {
                    float dx2 = other->x - a->x;
                    float dy2 = other->y - a->y;
                    if (dx2 > 0.5f) dx2 -= 1.0f;
                    if (dx2 < -0.5f) dx2 += 1.0f;
                    if (dy2 > 0.5f) dy2 -= 1.0f;
                    if (dy2 < -0.5f) dy2 += 1.0f;
                    float dist2 = sqrtf(dx2*dx2 + dy2*dy2);
                    if (dist2 < 0.1f) nearby_defenders++;
                }
            }
            value *= (1.0f + 0.2f * nearby_defenders);
            
            a->energy += value;
            a->fitness += value;
            r->collected = 1;
            
            // Deposit pheromone at resource location
            int ph_idx = get_ph_index(r->x, r->y);
            atomicAdd(&pheromones[ph_idx].strength[a->arch], PH_DEPOSIT);
        }
        
        // Communicate role: broadcast location to neighbors within 0.06
        if (a->role[2] > 0.3f) {
            for (int j = 0; j < AGENTS; j += AGENTS/32) { // sample
                if (j == idx) continue;
                Agent* other = &agents[j];
                if (other->arch == a->arch) {
                    float dx2 = other->x - a->x;
                    float dy2 = other->y - a->y;
                    if (dx2 > 0.5f) dx2 -= 1.0f;
                    if (dx2 < -0.5f) dx2 += 1.0f;
                    if (dy2 > 0.5f) dy2 -= 1.0f;
                    if (dy2 < -0.5f) dy2 += 1.0f;
                    float dist2 = sqrtf(dx2*dx2 + dy2*dy2);
                    if (dist2 < 0.06f) {
                        // Influence neighbor toward resource
                        float influence = a->role[2] * 0.01f;
                        other->vx += (r->x - other->x) * influence;
                        other->vy += (r->y - other->y) * influence;
                    }
                }
            }
        }
    }
    
    // Coupling: align with same archetype, avoid others
    for (int j = 0; j < AGENTS; j += AGENTS/64) { // sample
        if (j == idx) continue;
        Agent* other = &agents[j];
        
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist < 0.1f) {
            float coupling = (a->arch == other->arch) ? 0.02f : 0.002f;
            a->vx += (other->vx - a->vx) * coupling;
            a->vy += (other->vy - a->vy) * coupling;
        }
    }
}

int main() {
    // Allocate memory
    Agent* d_agents;
    Resource* d_resources;
    Pheromone* d_pheromones;
    
    cudaMalloc(&d_agents, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones, PH_GRID_SIZE * PH_GRID_SIZE * sizeof(Pheromone));
    
    // Initialize
    dim3 block(BLOCK_SIZE);
    dim3 grid_agents((AGENTS + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_res((RESOURCES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_ph((PH_GRID_SIZE*PH_GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    init_agents<<<grid_agents, block>>>(d_agents, d_pheromones);
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromones<<<grid_ph, block>>>(d_pheromones);
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        // Decay pheromones
        decay_pheromones<<<grid_ph, block>>>(d_pheromones);
        
        // Main tick
        tick<<<grid_agents, block>>>(d_agents, d_resources, d_pheromones, t);
        
        // Respawn resources every 50 ticks
        if (t % 50 == 49) {
            init_resources<<<grid_res, block>>>(d_resources);
        }
        
        cudaDeviceSynchronize();
    }
    
    // Copy results back