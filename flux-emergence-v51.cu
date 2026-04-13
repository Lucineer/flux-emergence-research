
/*
CUDA Simulation Experiment v51
Testing: STIGMERY TRAILS - agents deposit pheromones at resource locations
Prediction: Pheromones will enhance collective foraging efficiency, 
            giving specialists GREATER advantage (2.0x+) due to coordinated trails
Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
Comparison: Specialized (role[arch]=0.7) vs Uniform (all roles=0.25)
Expected: Specialists leverage trails better → higher fitness ratio
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
const float CELL_SIZE = WORLD_SIZE / PHEROMONE_GRID;

// Agent structure
struct Agent {
    float x, y;           // position
    float vx, vy;         // velocity
    float energy;         // energy level
    float role[4];        // behavioral roles: explore, collect, communicate, defend
    float fitness;        // accumulated fitness
    int arch;             // archetype 0-3
    unsigned int rng;     // random state
};

// Resource structure
struct Resource {
    float x, y;           // position
    float value;          // energy value
    int collected;        // 0=available, 1=collected
    unsigned int rng;     // random state for resource
};

// Pheromone structure for stigmergy
struct Pheromone {
    float strength[ARCHETYPES]; // pheromone strength per archetype
    float decay;                // decay rate
};

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return (lcg(state) & 0xFFFF) / 65535.0f;
}

// Kernel to initialize agents
__global__ void init_agents(Agent* agents, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    a->x = lcgf(&a->rng);
    a->y = lcgf(&a->rng);
    a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
    a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
    a->energy = 1.0f;
    a->fitness = 0.0f;
    a->arch = idx % ARCHETYPES;
    a->rng = idx * 12345 + 1;
    
    if (specialized) {
        // Specialized: primary role = 0.7, others = 0.1
        for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
        a->role[a->arch] = 0.7f;
    } else {
        // Uniform: all roles = 0.25
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
}

// Kernel to initialize resources
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    r->rng = idx * 67890 + 1;
    r->x = lcgf(&r->rng);
    r->y = lcgf(&r->rng);
    r->value = 0.5f + lcgf(&r->rng) * 0.5f; // 0.5-1.0
    r->collected = 0;
}

// Kernel to initialize pheromone grid
__global__ void init_pheromones(Pheromone* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    for (int i = 0; i < ARCHETYPES; i++) {
        grid[idx].strength[i] = 0.0f;
    }
    grid[idx].decay = 0.95f; // 5% decay per tick
}

// Kernel to decay pheromones
__global__ void decay_pheromones(Pheromone* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    for (int i = 0; i < ARCHETYPES; i++) {
        grid[idx].strength[i] *= grid[idx].decay;
    }
}

// Helper: get grid cell from position
__device__ int get_grid_cell(float x, float y) {
    int gx = (int)(x / CELL_SIZE) % PHEROMONE_GRID;
    int gy = (int)(y / CELL_SIZE) % PHEROMONE_GRID;
    return gy * PHEROMONE_GRID + gx;
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromones, 
                     int tick_num, float* fitness_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = (idx + 37) % AGENTS;
    Agent* other = &agents[other_idx];
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - other->role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Apply drift to non-dominant role
        int drift_role = (a->arch + 1) % 4;
        a->role[drift_role] += (lcgf(&a->rng) * 0.02f - 0.01f);
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) a->role[i] /= sum;
    }
    
    // Movement influenced by pheromones
    float move_x = a->vx;
    float move_y = a->vy;
    
    // Sample pheromone gradient in neighborhood
    int cell = get_grid_cell(a->x, a->y);
    float my_pheromone = pheromones[cell].strength[a->arch];
    
    // Check neighboring cells for stronger pheromones
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (int)(a->x / CELL_SIZE) + dx;
            int ny = (int)(a->y / CELL_SIZE) + dy;
            if (nx >= 0 && nx < PHEROMONE_GRID && ny >= 0 && ny < PHEROMONE_GRID) {
                int ncell = ny * PHEROMONE_GRID + nx;
                float neigh_pheromone = pheromones[ncell].strength[a->arch];
                if (neigh_pheromone > my_pheromone * 1.1f) {
                    // Move toward stronger pheromone
                    move_x += dx * 0.005f * a->role[0]; // explore role influences
                    move_y += dy * 0.005f * a->role[0];
                }
            }
        }
    }
    
    // Update position with velocity limit
    float speed = sqrtf(move_x * move_x + move_y * move_y);
    if (speed > 0.02f) {
        move_x = move_x / speed * 0.02f;
        move_y = move_y / speed * 0.02f;
    }
    
    a->x += move_x;
    a->y += move_y;
    
    // World boundaries (toroidal)
    if (a->x < 0) a->x += 1.0f;
    if (a->x >= 1.0f) a->x -= 1.0f;
    if (a->y < 0) a->y += 1.0f;
    if (a->y >= 1.0f) a->y -= 1.0f;
    
    // Resource collection
    float collect_range = 0.02f + a->role[1] * 0.02f; // collect role increases range
    float best_value = 0.0f;
    int best_res = -1;
    
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
        
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist < collect_range && r->value > best_value) {
            best_value = r->value;
            best_res = i;
        }
    }
    
    if (best_res != -1) {
        Resource* r = &resources[best_res];
        
        // Territory bonus: defenders nearby
        float territory_bonus = 1.0f;
        int defenders_nearby = 0;
        for (int i = 0; i < AGENTS; i += AGENTS/16) { // sample 16 agents
            Agent* other = &agents[i];
            if (other->arch == a->arch && other != a) {
                float dx = other->x - a->x;
                float dy = other->y - a->y;
                if (dx > 0.5f) dx -= 1.0f;
                if (dx < -0.5f) dx += 1.0f;
                if (dy > 0.5f) dy -= 1.0f;
                if (dy < -0.5f) dy += 1.0f;
                float dist = sqrtf(dx * dx + dy * dy);
                if (dist < 0.06f && other->role[3] > 0.3f) {
                    defenders_nearby++;
                }
            }
        }
        territory_bonus += defenders_nearby * 0.2f;
        
        // Collect bonus from collect role
        float collect_bonus = 1.0f + a->role[1] * 0.5f;
        
        float gained = r->value * territory_bonus * collect_bonus;
        a->energy += gained;
        a->fitness += gained;
        
        // DEPOSIT PHEROMONE at resource location (STIGMERY)
        int pcell = get_grid_cell(r->x, r->y);
        atomicAdd(&pheromones[pcell].strength[a->arch], 0.5f);
        
        r->collected = 1;
        
        // Communicate to nearby agents (communicate role)
        if (a->role[2] > 0.3f) {
            for (int i = 0; i < AGENTS; i += AGENTS/32) { // sample 32 agents
                Agent* other = &agents[i];
                if (other->arch == a->arch && other != a) {
                    float dx = other->x - a->x;
                    float dy = other->y - a->y;
                    if (dx > 0.5f) dx -= 1.0f;
                    if (dx < -0.5f) dx += 1.0f;
                    if (dy > 0.5f) dy -= 1.0f;
                    if (dy < -0.5f) dy += 1.0f;
                    float dist = sqrtf(dx * dx + dy * dy);
                    if (dist < 0.06f) {
                        // Influence other's movement toward resource
                        other->vx += (r->x - other->x) * 0.01f * a->role[2];
                        other->vy += (r->y - other->y) * 0.01f * a->role[2];
                    }
                }
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0 && lcgf(&a->rng) < 0.3f) {
        // Defenders resist perturbation
        if (a->role[3] < 0.5f) {
            a->energy *= 0.5f;
            a->vx = lcgf(&a->rng) * 0.04f - 0.02f;
            a->vy = lcgf(&a->rng) * 0.04f - 0.02f;
        }
    }
    
    // Coupling: adjust roles toward same archetype neighbors
    int coupled_idx = (idx + 19) % AGENTS;
    Agent* coupled = &agents[coupled_idx];
    float coupling_strength = (a->arch == coupled->arch) ? 0.02f : 0.002f;
    
    for (int i = 0; i < 4; i++) {
        a->role[i] += (coupled->role[i] - a->role[i]) * coupling_strength;
    }
    
    // Renormalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) a->role[i] /= sum;
    
    // Update fitness sum for this archetype
    atomicAdd(&fitness_sum[a->arch], a->fitness);
}

// Kernel to respawn resources
__global__ void respawn_resources(Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    // Respawn every 50 ticks
    if (tick_num % 50 == 0) {
        if (r->collected || lcgf(&r->rng) < 0.3f) {
            r->x = lcgf(&r->rng);
            r->y = lcgf(&r->rng);
            r->value = 0.5f + lcgf(&r->rng) * 0.5f;
            r->collected = 0;
        }
    }
}

int main() {
    printf("Experiment v51: Stigmergy Trails\n");
    printf("Testing if pheromone trails enhance specialist advantage\n");
    printf("Prediction: Specialists 2.0x+ better than uniform\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RESOURCES, TICKS);
    
    // Allocate device memory
    Agent* d_agents_spec;
    Agent* d_agents_unif;
    Resource* d_resources;
    Pheromone* d_pheromones_spec;
    Pheromone* d_pheromones_unif;
    float* d_fitness_spec;
    float* d_fitness_unif;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_unif, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, PHEROMONE_GRID * PHEROMONE_GRID * sizeof(Pheromone));
    cudaMalloc(&d_pheromones_unif, PHEROMONE_GRID * PHEROMONE_GRID * sizeof(Pheromone));
    cudaMalloc(&d_fitness_spec, ARCHETYPES * sizeof(float));
    cudaMalloc(&d_fitness_unif, ARCHETYPES * sizeof(float));
    
    // Initialize
    dim3 block(256);
    dim3 grid_spec((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONE_GRID * PHEROMONE_GRID + 255) / 256);
    
    // Specialized population
    init_agents<<<grid_spec, block>>>(d_agents_spec, 1);
    init_pheromones<<<grid_ph, block>>