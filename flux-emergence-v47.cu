// CUDA Simulation Experiment v47: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist efficiency
// Prediction: Pheromones will help uniform agents more than specialists (reducing specialist advantage)
// because uniform agents can follow trails without role coordination overhead
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence
// Novel: Agents leave pheromone markers at collected resources that decay over time
// Other agents can detect pheromone intensity and move toward strongest signal

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RES_COUNT = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Pheromone grid constants
const int GRID_SIZE = 256;
const float CELL_SIZE = 1.0f / GRID_SIZE;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_STRENGTH = 0.5f;
const float PHEROMONE_DETECTION_RANGE = 0.08f;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3, ARCH_COUNT = 4 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role strengths: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Dominant archetype
    unsigned int rng;     // Random state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
    unsigned int rng;     // Random state for initialization
};

// Pheromone structure for stigmergy
struct Pheromone {
    float intensity[ARCH_COUNT];  // Pheromone intensity per archetype
    float x, y;                   // Grid cell center position
};

// Linear Congruential Generator (LCG)
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize pheromone grid
__global__ void initPheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE * GRID_SIZE) return;
    
    int i = idx % GRID_SIZE;
    int j = idx / GRID_SIZE;
    
    pheromones[idx].x = (i + 0.5f) * CELL_SIZE;
    pheromones[idx].y = (j + 0.5f) * CELL_SIZE;
    for (int k = 0; k < ARCH_COUNT; k++) {
        pheromones[idx].intensity[k] = 0.0f;
    }
}

// Decay pheromones over time
__global__ void decayPheromones(Pheromone* pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE * GRID_SIZE) return;
    
    for (int k = 0; k < ARCH_COUNT; k++) {
        pheromones[idx].intensity[k] *= PHEROMONE_DECAY;
    }
}

// Initialize agents
__global__ void initAgents(Agent* agents, Pheromone* pheromones, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    agents[idx].x = lcgf(&agents[idx].rng);
    agents[idx].y = lcgf(&agents[idx].rng);
    agents[idx].vx = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].rng = idx * 12345 + 6789;
    
    if (specialized) {
        // Specialized agents: one dominant role (0.7), others 0.1 each
        agents[idx].arch = idx % ARCH_COUNT;
        for (int i = 0; i < ARCH_COUNT; i++) {
            agents[idx].role[i] = (i == agents[idx].arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles equal
        agents[idx].arch = ARCH_EXPLORER;
        for (int i = 0; i < ARCH_COUNT; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void initResources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES_COUNT) return;
    
    resources[idx].rng = idx * 54321 + 9876;
    resources[idx].x = lcgf(&resources[idx].rng) * 0.9f + 0.05f;
    resources[idx].y = lcgf(&resources[idx].rng) * 0.9f + 0.05f;
    resources[idx].value = 0.5f + lcgf(&resources[idx].rng) * 0.5f;
    resources[idx].collected = 0;
}

// Find grid cell for position
__device__ int getGridCell(float x, float y) {
    int xi = min(max((int)(x / CELL_SIZE), 0), GRID_SIZE - 1);
    int yi = min(max((int)(y / CELL_SIZE), 0), GRID_SIZE - 1);
    return yi * GRID_SIZE + xi;
}

// Deposit pheromone at location
__device__ void depositPheromone(Pheromone* pheromones, float x, float y, int arch) {
    int cell = getGridCell(x, y);
    atomicAdd(&pheromones[cell].intensity[arch], PHEROMONE_STRENGTH);
}

// Get pheromone gradient direction
__device__ void getPheromoneGradient(Pheromone* pheromones, float x, float y, int arch, 
                                     float* grad_x, float* grad_y) {
    int cell = getGridCell(x, y);
    float center = pheromones[cell].intensity[arch];
    
    // Check neighboring cells
    float max_val = center;
    int max_cell = cell;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = (int)(x / CELL_SIZE) + dx;
            int ny = (int)(y / CELL_SIZE) + dy;
            
            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
                int ncell = ny * GRID_SIZE + nx;
                float val = pheromones[ncell].intensity[arch];
                if (val > max_val) {
                    max_val = val;
                    max_cell = ncell;
                }
            }
        }
    }
    
    if (max_cell != cell && max_val > 0.01f) {
        *grad_x = pheromones[max_cell].x - x;
        *grad_y = pheromones[max_cell].y - y;
        float len = sqrtf(*grad_x * *grad_x + *grad_y * *grad_y);
        if (len > 0.0f) {
            *grad_x /= len;
            *grad_y /= len;
        }
    } else {
        *grad_x = 0.0f;
        *grad_y = 0.0f;
    }
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, Pheromone* pheromones, 
                     int tick_num, int specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence mechanism (v3)
    float similarity = 0.0f;
    for (int i = 0; i < ARCH_COUNT; i++) {
        similarity += fabsf(a->role[i] - 0.25f);
    }
    similarity /= ARCH_COUNT;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant role
        int drift_role = (int)(lcgf(&a->rng) * (ARCH_COUNT - 1));
        if (drift_role >= a->arch) drift_role++;
        
        float drift = lcgf(&a->rng) * 0.02f - 0.01f;
        a->role[drift_role] = max(0.0f, min(1.0f, a->role[drift_role] + drift));
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < ARCH_COUNT; i++) sum += a->role[i];
        for (int i = 0; i < ARCH_COUNT; i++) a->role[i] /= sum;
    }
    
    // Role-based behavior with pheromone following
    float explore_strength = a->role[ARCH_EXPLORER];
    float collect_strength = a->role[ARCH_COLLECTOR];
    float comm_strength = a->role[ARCH_COMMUNICATOR];
    float defend_strength = a->role[ARCH_DEFENDER];
    
    // Pheromone influence (novel mechanism)
    float pheromone_grad_x = 0.0f, pheromone_grad_y = 0.0f;
    if (explore_strength > 0.3f || collect_strength > 0.3f) {
        getPheromoneGradient(pheromones, a->x, a->y, a->arch, 
                            &pheromone_grad_x, &pheromone_grad_y);
    }
    
    // Movement with pheromone influence
    float move_x = a->vx;
    float move_y = a->vy;
    
    // Add pheromone following if gradient exists
    if (pheromone_grad_x != 0.0f || pheromone_grad_y != 0.0f) {
        float influence = min(explore_strength + collect_strength, 1.0f) * 0.3f;
        move_x = move_x * (1.0f - influence) + pheromone_grad_x * influence;
        move_y = move_y * (1.0f - influence) + pheromone_grad_y * influence;
    }
    
    // Random exploration component
    move_x += (lcgf(&a->rng) - 0.5f) * 0.01f * explore_strength;
    move_y += (lcgf(&a->rng) - 0.5f) * 0.01f * explore_strength;
    
    // Normalize velocity
    float speed = sqrtf(move_x * move_x + move_y * move_y);
    if (speed > 0.02f) {
        move_x = move_x / speed * 0.02f;
        move_y = move_y / speed * 0.02f;
    }
    
    a->vx = move_x;
    a->vy = move_y;
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary check
    if (a->x < 0.0f) { a->x = 0.0f; a->vx = fabsf(a->vx); }
    if (a->x > 1.0f) { a->x = 1.0f; a->vx = -fabsf(a->vx); }
    if (a->y < 0.0f) { a->y = 0.0f; a->vy = fabsf(a->vy); }
    if (a->y > 1.0f) { a->y = 1.0f; a->vy = -fabsf(a->vy); }
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    // Explorer detection range
    float detect_range = 0.03f + explore_strength * 0.04f;
    
    for (int i = 0; i < RES_COUNT; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx * dx + dy * dy);
        
        if (dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
        
        // Detection
        if (dist < detect_range) {
            // Collector grab range with bonus
            float grab_range = 0.02f + collect_strength * 0.02f;
            if (dist < grab_range) {
                // Collection with value bonus for collectors
                float bonus = 1.0f + collect_strength * 0.5f;
                a->energy += r->value * bonus;
                a->fitness += r->value * bonus;
                r->collected = 1;
                
                // Deposit pheromone at resource location (novel mechanism)
                depositPheromone(pheromones, r->x, r->y, a->arch);
                
                // Defender territory boost
                float defender_boost = 1.0f;
                for (int j = 0; j < AGENT_COUNT; j++) {
                    if (j == idx) continue;
                    Agent* other = &agents[j];
                    if (other->arch == ARCH_DEFENDER && a->arch == ARCH_DEFENDER) {
                        float odx = other->x - a->x;
                        float ody = other->y - a->y;
                        if (sqrtf(odx * odx + ody * ody) < 0.1f) {
                            defender_boost += 0.2f;
                        }
                    }
                }
                a->energy *= defender_boost;
                a->fitness *= defender_boost;
                
                break;
            }
            
            // Communication broadcast
            if (comm_strength > 0.3f) {
                for (int j = 0; j < AGENT_COUNT; j++) {
                    if (j == idx) continue;
                    Agent* other = &agents[j];
                    float odx = other->x - a->x;
                    float ody = other->y - a->y;
                    if (sqrtf(odx * odx + ody * ody) < 0.06f) {
                        // Attract neighbor toward resource
                        float influence = 0.1f * comm_strength;
                        other->vx += (r->x - other->x) * influence;
                        other->vy += (r->y - other->y) * influence;
                    }
                }
            }
        }
    }
    
    // Perturbation resistance for defenders
    if (tick_num % 100 == 0 && defend_strength < 0.5f) {
        a->energy *= 0.5f;
    }
    
    // Update archetype based on strongest role
    float max_role = a->role[0];
    int max_arch = 0;
    for (int i = 1; i < ARCH_COUNT; i++) {
        if (a->role[i] > max_role) {
            max_role = a->role[i];
            max_arch = i;
        }
    }
    a->arch = max_arch;
}

int main() {
    printf("Experiment v47: Stigmergy with Pheromone Trails\n");
    printf("Testing: Whether pheromones reduce specialist advantage\n");
    printf("Prediction: Uniform +30%% vs specialists with pheromones\n");
    printf("Mechanisms: Scarcity(128res), Territory, Comms + Anti-convergence + Pheromones\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENT_COUNT, RES_COUNT, TICKS);
    
    // Allocate device memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources;
    Pheromone* d_pheromones_spec;
    Pheromone* d_pheromones_uniform;
    
    cudaMalloc(&d_agents_spec, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_resources, RES_COUNT * sizeof(Resource));
    cudaMalloc(&d_pheromones_spec, GRID_SIZE * GRID_SIZE * sizeof(Pheromone));
    cudaMalloc(&d_pheromones_uniform, GRID_SIZE *