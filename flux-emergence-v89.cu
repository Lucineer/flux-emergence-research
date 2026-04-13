// CUDA Simulation Experiment v89: Stigmergy with Pheromone Trails
// Testing: Agents leave pheromone markers at resource locations that decay over time
// Prediction: Pheromones will enhance specialist coordination, increasing advantage ratio >1.61x
// Novel mechanism: Stigmergy (indirect communication through environment modification)
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Control: Uniform agents (all roles=0.25) vs Specialized (role[arch]=0.7)

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 64; // 64x64 grid for pheromone field
const int PHEROMONE_CAPACITY = 16; // Max pheromones per cell

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles
    float fitness;        // Fitness score
    int arch;             // Dominant archetype
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
};

// Pheromone structure
struct Pheromone {
    float x, y;           // Location
    float strength;       // Strength (0-1)
    int arch;             // Archetype that left it
    int age;              // Age in ticks
};

// Pheromone grid cell
struct PheromoneCell {
    Pheromone pheromones[PHEROMONE_CAPACITY];
    int count;
};

// LCG RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents kernel
__global__ void init_agents(Agent* agents, Resource* resources, PheromoneCell* pheromone_grid) {
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
    
    // Specialized vs uniform groups (first half specialized, second half uniform)
    if (idx < AGENTS/2) {
        // Specialized agents: one dominant role
        a->arch = idx % 4;
        for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
        a->role[a->arch] = 0.7f;
    } else {
        // Uniform control agents: all roles equal
        a->arch = -1;
        for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
    }
}

// Initialize resources kernel
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    unsigned int seed = idx * 19 + 54321;
    r->x = (seed * 1664525u + 1013904223u) / 4294967296.0f;
    r->y = ((seed+1) * 1664525u + 1013904223u) / 4294967296.0f;
    r->value = 0.5f + (seed % 100) / 200.0f;
    r->collected = 0;
}

// Initialize pheromone grid
__global__ void init_pheromones(PheromoneCell* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    grid[idx].count = 0;
}

// Update pheromones (decay and remove old)
__global__ void update_pheromones(PheromoneCell* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONE_GRID * PHEROMONE_GRID) return;
    
    PheromoneCell* cell = &grid[idx];
    int write_idx = 0;
    
    for (int i = 0; i < cell->count; i++) {
        cell->pheromones[i].age++;
        cell->pheromones[i].strength *= 0.95f; // Decay
        
        // Keep if still strong enough and not too old
        if (cell->pheromones[i].strength > 0.01f && cell->pheromones[i].age < 100) {
            cell->pheromones[write_idx++] = cell->pheromones[i];
        }
    }
    cell->count = write_idx;
}

// Add pheromone at location
__device__ void add_pheromone(PheromoneCell* grid, float x, float y, int arch) {
    int gx = min(PHEROMONE_GRID-1, max(0, (int)(x * PHEROMONE_GRID)));
    int gy = min(PHEROMONE_GRID-1, max(0, (int)(y * PHEROMONE_GRID)));
    int cell_idx = gy * PHEROMONE_GRID + gx;
    
    PheromoneCell* cell = &grid[cell_idx];
    if (cell->count < PHEROMONE_CAPACITY) {
        Pheromone* p = &cell->pheromones[cell->count++];
        p->x = x;
        p->y = y;
        p->strength = 1.0f;
        p->arch = arch;
        p->age = 0;
    }
}

// Sample pheromones in area
__device__ float sample_pheromones(PheromoneCell* grid, float x, float y, float radius, int arch) {
    int min_gx = max(0, (int)((x - radius) * PHEROMONE_GRID));
    int max_gx = min(PHEROMONE_GRID-1, (int)((x + radius) * PHEROMONE_GRID));
    int min_gy = max(0, (int)((y - radius) * PHEROMONE_GRID));
    int max_gy = min(PHEROMONE_GRID-1, (int)((y + radius) * PHEROMONE_GRID));
    
    float total = 0.0f;
    for (int gy = min_gy; gy <= max_gy; gy++) {
        for (int gx = min_gx; gx <= max_gx; gx++) {
            PheromoneCell* cell = &grid[gy * PHEROMONE_GRID + gx];
            for (int i = 0; i < cell->count; i++) {
                Pheromone* p = &cell->pheromones[i];
                float dx = p->x - x;
                float dy = p->y - y;
                float dist2 = dx*dx + dy*dy;
                if (dist2 < radius*radius) {
                    // Same archetype pheromones are more attractive
                    float weight = p->strength * (p->arch == arch ? 1.5f : 0.5f);
                    total += weight / (1.0f + dist2 * 100.0f);
                }
            }
        }
    }
    return total;
}

// Main simulation tick kernel
__global__ void tick(Agent* agents, Resource* resources, PheromoneCell* pheromone_grid, 
                     int tick_num, int* perturbations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: random drift when too similar to neighbors
    if (tick_num % 10 == 0) {
        int neighbor_idx = (idx + 1) % AGENTS;
        Agent* neighbor = &agents[neighbor_idx];
        
        float similarity = 0.0f;
        for (int i = 0; i < 4; i++) {
            similarity += fabsf(a->role[i] - neighbor->role[i]);
        }
        similarity = 1.0f - similarity / 4.0f;
        
        if (similarity > 0.9f) {
            int drift_role = (a->arch + 1) % 4; // Drift non-dominant role
            a->role[drift_role] += (lcgf(&a->rng) - 0.5f) * 0.01f;
            
            // Renormalize
            float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
            for (int i = 0; i < 4; i++) a->role[i] /= sum;
        }
    }
    
    // Pheromone sensing (NOVEL MECHANISM)
    float pheromone_force_x = 0.0f;
    float pheromone_force_y = 0.0f;
    float sense_radius = 0.05f;
    
    // Sample pheromone gradient
    float center = sample_pheromones(pheromone_grid, a->x, a->y, sense_radius, a->arch);
    float dx = sample_pheromones(pheromone_grid, a->x + 0.01f, a->y, sense_radius, a->arch) - center;
    float dy = sample_pheromones(pheromone_grid, a->x, a->y + 0.01f, sense_radius, a->arch) - center;
    
    // Move toward pheromone gradient (explorers and communicators are more sensitive)
    float pheromone_sensitivity = a->role[ARCH_EXPLORER] + a->role[ARCH_COMMUNICATOR];
    pheromone_force_x = dx * pheromone_sensitivity * 0.5f;
    pheromone_force_y = dy * pheromone_sensitivity * 0.5f;
    
    // Role-based behavior
    float move_x = 0.0f, move_y = 0.0f;
    
    // Explorer: random exploration with some pheromone following
    if (a->role[ARCH_EXPLORER] > 0.3f) {
        move_x += (lcgf(&a->rng) - 0.5f) * 0.02f * a->role[ARCH_EXPLORER];
        move_y += (lcgf(&a->rng) - 0.5f) * 0.02f * a->role[ARCH_EXPLORER];
    }
    
    // Collector: seek resources
    float nearest_dist = 1.0f;
    float nearest_x = 0.0f, nearest_y = 0.0f;
    int nearest_idx = -1;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        float detect_range = 0.03f + 0.04f * a->role[ARCH_EXPLORER];
        if (dist < detect_range && dist < nearest_dist) {
            nearest_dist = dist;
            nearest_x = r->x;
            nearest_y = r->y;
            nearest_idx = i;
        }
    }
    
    if (nearest_idx >= 0) {
        // Move toward resource
        float dx = nearest_x - a->x;
        float dy = nearest_y - a->y;
        float dist = max(0.001f, sqrtf(dx*dx + dy*dy));
        move_x += (dx / dist) * 0.01f * a->role[ARCH_COLLECTOR];
        move_y += (dy / dist) * 0.01f * a->role[ARCH_COLLECTOR];
        
        // Try to collect
        float grab_range = 0.02f + 0.02f * a->role[ARCH_COLLECTOR];
        if (dist < grab_range) {
            Resource* r = &resources[nearest_idx];
            if (!r->collected) {
                r->collected = 1;
                float value = r->value * (1.0f + 0.5f * a->role[ARCH_COLLECTOR]); // Collector bonus
                
                // Defender territory bonus
                int defenders_nearby = 0;
                for (int j = 0; j < AGENTS; j += AGENTS/16) { // Sample
                    Agent* other = &agents[j];
                    if (other == a) continue;
                    float odx = other->x - a->x;
                    float ody = other->y - a->y;
                    if (odx*odx + ody*ody < 0.04f && other->role[ARCH_DEFENDER] > 0.5f) {
                        defenders_nearby++;
                    }
                }
                value *= (1.0f + 0.2f * defenders_nearby);
                
                a->energy += value;
                a->fitness += value;
                
                // Leave pheromone at resource location (NOVEL MECHANISM)
                add_pheromone(pheromone_grid, r->x, r->y, a->arch);
            }
        }
    }
    
    // Communicator: broadcast and move toward pheromones
    if (a->role[ARCH_COMMUNICATOR] > 0.3f) {
        // Enhanced pheromone following for communicators
        pheromone_force_x *= 2.0f;
        pheromone_force_y *= 2.0f;
    }
    
    // Defender: resist perturbations and stay near pheromones
    if (tick_num % 50 == 0 && lcgf(&a->rng) < 0.1f) {
        if (a->role[ARCH_DEFENDER] < 0.3f) {
            a->energy *= 0.5f; // Perturbation
            atomicAdd(perturbations, 1);
        }
    }
    
    // Apply movement with pheromone influence
    a->vx = a->vx * 0.9f + move_x + pheromone_force_x;
    a->vy = a->vy * 0.9f + move_y + pheromone_force_y;
    
    // Update position with bounds
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0.0f) { a->x = 0.0f; a->vx *= -0.5f; }
    if (a->x > 1.0f) { a->x = 1.0f; a->vx *= -0.5f; }
    if (a->y < 0.0f) { a->y = 0.0f; a->vy *= -0.5f; }
    if (a->y > 1.0f) { a->y = 1.0f; a->vy *= -0.5f; }
    
    // Respawn resources periodically
    if (tick_num % 50 == 0) {
        for (int i = 0; i < RESOURCES; i++) {
            if (resources[i].collected && lcgf(&a->rng) < 0.3f) {
                resources[i].collected = 0;
                resources[i].x = lcgf(&a->rng);
                resources[i].y = lcgf(&a->rng);
            }
        }
    }
}

int main() {
    printf("Experiment v89: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone markers enhance specialist coordination\n");
    printf("Prediction: Specialist advantage ratio > 1.61x (baseline)\n");
    printf("Agents: %d (512 specialized, 512 uniform control)\n", AGENTS);
    printf("Resources: %d, Ticks: %d\n", RESOURCES, TICKS);
    printf("Pheromone grid: %dx%d\n\n", PHEROMONE_GRID, PHEROMONE_GRID);
    
    // Allocate device memory
    Agent* d_agents;
    Resource* d_resources;
   