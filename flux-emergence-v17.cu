
/*
CUDA Simulation Experiment v17: STIGMERY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage
            over uniform agents by >1.61x (v8 baseline) since specialists can follow
            trails left by same-arch agents.
Baseline: v8 mechanisms (scarcity, territory, comms) included.
Novelty: Stigmergy - agents deposit pheromone at collected resources, all agents can
         sense pheromone gradient and move toward stronger concentrations.
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

// Agent structure
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];      // role[0]=explore,1=collect,2=communicate,3=defend
    float fitness;
    int arch;           // archetype 0-3
    unsigned int rng;   // random state
};

// Resource structure
struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone grid cell
struct Pheromone {
    float strength[ARCHETYPES];
};

// Linear congruential generator (device/host)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, Pheromone* grid, int grid_size) {
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
    
    // Specialized agents (first half) vs uniform control (second half)
    if (idx < AGENTS/2) {
        // Specialized: high value in own archetype role
        for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
        a->role[a->arch] = 0.7f;
    } else {
        // Uniform control: all roles equal
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
    
    // Normalize
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) a->role[i] /= sum;
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    unsigned int seed = idx * 19 + 54321;
    r->x = (seed * 1103515245 + 12345) / 4294967296.0f;
    r->y = ((seed * 1103515245 + 12345) * 1103515245 + 12345) / 4294967296.0f;
    r->value = 0.8f + (seed % 100) / 500.0f;
    r->collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromone(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    Pheromone* p = &grid[idx];
    for (int i = 0; i < ARCHETYPES; i++) {
        p->strength[i] = 0.0f;
    }
}

// Decay pheromones kernel
__global__ void decay_pheromone(Pheromone* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size * grid_size) return;
    
    Pheromone* p = &grid[idx];
    for (int i = 0; i < ARCHETYPES; i++) {
        p->strength[i] *= 0.95f;  // 5% decay per tick
    }
}

// Main simulation kernel
__global__ void simulation_tick(Agent* agents, Resource* resources, Pheromone* grid, 
                     int grid_size, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect similarity > 0.9 with neighbors
    int similar_count = 0;
    for (int i = 0; i < 10; i++) {
        int j = (idx + i * 103) % AGENTS;
        if (j == idx) continue;
        Agent* other = &agents[j];
        float dx = a->x - other->x;
        float dy = a->y - other->y;
        if (dx*dx + dy*dy < 0.04f) {  // 0.2 radius
            float similarity = 0.0f;
            for (int k = 0; k < 4; k++) {
                similarity += fminf(a->role[k], other->role[k]);
            }
            if (similarity > 0.9f) similar_count++;
        }
    }
    
    // Apply anti-convergence drift
    if (similar_count > 2) {
        int drift_role = (a->arch + tick_num) % 4;
        if (drift_role != a->arch) {
            a->role[drift_role] += 0.01f;
            // Renormalize
            float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
            for (int i = 0; i < 4; i++) a->role[i] /= sum;
        }
    }
    
    // Perturbation every 50 ticks (defenders resist)
    if (tick_num % 50 == 0) {
        if (a->role[3] < 0.3f) {  // Not a defender
            a->energy *= 0.5f;
        }
    }
    
    // Sense pheromone gradient (NOVEL MECHANISM)
    int gx = (int)(a->x * grid_size);
    int gy = (int)(a->y * grid_size);
    float pheromone_x = 0.0f, pheromone_y = 0.0f;
    
    // Sample 3x3 grid around agent
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int sx = gx + dx;
            int sy = gy + dy;
            if (sx >= 0 && sx < grid_size && sy >= 0 && sy < grid_size) {
                Pheromone* p = &grid[sy * grid_size + sx];
                float strength = p->strength[a->arch];
                pheromone_x += strength * dx;
                pheromone_y += strength * dy;
            }
        }
    }
    
    // Normalize pheromone influence
    float pnorm = sqrtf(pheromone_x*pheromone_x + pheromone_y*pheromone_y);
    if (pnorm > 1e-6f) {
        pheromone_x /= pnorm;
        pheromone_y /= pnorm;
    }
    
    // Role-based behavior with pheromone influence
    float explore_dir_x = 0.0f, explore_dir_y = 0.0f;
    float collect_dir_x = 0.0f, collect_dir_y = 0.0f;
    float comm_dir_x = 0.0f, comm_dir_y = 0.0f;
    
    // Explore behavior: random walk + pheromone following
    explore_dir_x = (lcgf(&a->rng) - 0.5f) * 0.1f + pheromone_x * 0.05f;
    explore_dir_y = (lcgf(&a->rng) - 0.5f) * 0.1f + pheromone_y * 0.05f;
    
    // Find nearest resource
    int nearest_res = -1;
    float min_dist2 = 1e6f;
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist2 = dx*dx + dy*dy;
        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            nearest_res = i;
        }
    }
    
    // Collect behavior: move toward nearest resource
    if (nearest_res >= 0) {
        Resource* r = &resources[nearest_res];
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        if (dist > 1e-6f) {
            collect_dir_x = dx / dist * 0.08f;
            collect_dir_y = dy / dist * 0.08f;
        }
    }
    
    // Communicate behavior: move toward other agents
    int comm_target = (idx + 1) % AGENTS;
    Agent* other = &agents[comm_target];
    float cdx = other->x - a->x;
    float cdy = other->y - a->y;
    float cdist = sqrtf(cdx*cdx + cdy*cdy);
    if (cdist > 1e-6f) {
        comm_dir_x = cdx / cdist * 0.06f;
        comm_dir_y = cdy / cdist * 0.06f;
    }
    
    // Combine behaviors based on role strengths
    a->vx = a->role[0] * explore_dir_x + 
            a->role[1] * collect_dir_x + 
            a->role[2] * comm_dir_x;
    a->vy = a->role[0] * explore_dir_y + 
            a->role[1] * collect_dir_y + 
            a->role[2] * comm_dir_y;
    
    // Normalize velocity
    float vnorm = sqrtf(a->vx*a->vx + a->vy*a->vy);
    if (vnorm > 0.02f) {
        a->vx = a->vx / vnorm * 0.02f;
        a->vy = a->vy / vnorm * 0.02f;
    }
    
    // Update position with wrap-around
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0.0f) a->x += 1.0f;
    if (a->x >= 1.0f) a->x -= 1.0f;
    if (a->y < 0.0f) a->y += 1.0f;
    if (a->y >= 1.0f) a->y -= 1.0f;
    
    // Resource collection
    if (nearest_res >= 0) {
        Resource* r = &resources[nearest_res];
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist2 = dx*dx + dy*dy;
        
        float detect_range = 0.05f + a->role[0] * 0.02f;  // Explore role helps detection
        float grab_range = 0.03f + a->role[1] * 0.01f;    // Collect role helps grabbing
        
        if (dist2 < detect_range*detect_range) {
            // Communicate resource location to nearby agents
            for (int i = 0; i < 5; i++) {
                int j = (idx + i * 67) % AGENTS;
                if (j == idx) continue;
                Agent* neighbor = &agents[j];
                float ndx = neighbor->x - a->x;
                float ndy = neighbor->y - a->y;
                if (ndx*ndx + ndy*ndy < 0.06f*0.06f && a->role[2] > 0.2f) {
                    // Communication happens
                }
            }
            
            // Collect resource
            if (dist2 < grab_range*grab_range && !r->collected) {
                r->collected = 1;
                float bonus = 1.0f + a->role[1] * 0.5f;  // Collect bonus
                
                // Territory bonus from nearby defenders
                float defend_bonus = 1.0f;
                for (int i = 0; i < 5; i++) {
                    int j = (idx + i * 89) % AGENTS;
                    Agent* neighbor = &agents[j];
                    if (neighbor->arch == a->arch && neighbor->role[3] > 0.3f) {
                        float ddx = neighbor->x - a->x;
                        float ddy = neighbor->y - a->y;
                        if (ddx*ddx + ddy*ddy < 0.1f*0.1f) {
                            defend_bonus += 0.2f;
                        }
                    }
                }
                
                float gain = r->value * bonus * defend_bonus;
                a->energy += gain;
                a->fitness += gain;
                
                // DEPOSIT PHEROMONE AT COLLECTION SITE (NOVEL MECHANISM)
                int px = (int)(r->x * grid_size);
                int py = (int)(r->y * grid_size);
                if (px >= 0 && px < grid_size && py >= 0 && py < grid_size) {
                    atomicAdd(&grid[py * grid_size + px].strength[a->arch], 1.0f);
                }
            }
        }
    }
    
    // Energy coupling with same/different archetypes
    for (int i = 0; i < 3; i++) {
        int j = (idx + i * 151) % AGENTS;
        Agent* other = &agents[j];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        if (dx*dx + dy*dy < 0.04f) {
            float coupling = (a->arch == other->arch) ? 0.02f : 0.002f;
            a->energy += coupling * (other->energy - a->energy);
        }
    }
}

int main() {
    // Allocate host memory
    Agent* h_agents = new Agent[AGENTS];
    Resource* h_resources = new Resource[RESOURCES];
    
    // Allocate device memory
    Agent* d_agents;
    Resource* d_resources;
    cudaMalloc(&d_agents, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources, sizeof(Resource) * RESOURCES);
    
    // Initialize pheromone grid (64x64)
    const int GRID_SIZE = 64;
    Pheromone* d_grid;
    cudaMalloc(&d_grid, sizeof(Pheromone) * GRID_SIZE * GRID_SIZE);
    
    // Initialize agents
    dim3 block(BLOCK_SIZE);
    dim3 grid_agents((AGENTS + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_res((RESOURCES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_ph((GRID_SIZE*GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    init_agents<<<grid_agents, block>>>(d_agents, d_grid, GRID_SIZE);
    init_resources<<<grid_res, block>>>(d_resources);
    init_pheromone<<<grid_ph, block>>>(d_grid, GRID_SIZE);
    cudaDeviceSynchronize();
    
    // Main simulation loop
    for (int tick = 0; tick < TICKS; tick++) {
        // Decay pheromones
        decay_pheromone<<<grid_ph, block>>>(d_grid, GRID_SIZE);
        
        // Run tick
        simulation_tick<<<grid_agents, block>>>(d_agents, d_resources, d_grid, GRID_SIZE, tick);
        cudaDeviceSynchronize();
        
        // Respawn resources every 50 ticks
        if (tick % 50 == 49) {
            init_resources<<