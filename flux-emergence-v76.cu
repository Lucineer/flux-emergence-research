// CUDA Simulation Experiment v76: Stigmergy with Pheromone Trails
// Testing: Whether pheromone trails left at resource locations improve specialist efficiency
// Prediction: Pheromones will help uniform agents more (they explore randomly), 
//             reducing specialist advantage from 1.61x to ~1.3x
// Mechanism: Agents deposit pheromone at collected resources, others follow gradient
// Baseline: v8 confirmed mechanisms (scarcity, territory, comms) + anti-convergence

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID = 64; // 64x64 grid for pheromone map
const float WORLD_SIZE = 1.0f;
const float CELL_SIZE = WORLD_SIZE / PHEROMONE_GRID;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3 };

// Agent struct
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role weights: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Dominant archetype
    unsigned int rng;     // RNG state
};

// Resource struct
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
    unsigned int rng;     // RNG state for initialization
};

// Pheromone grid (device global memory)
__device__ float d_pheromone[PHEROMONE_GRID][PHEROMONE_GRID];
__device__ int d_phero_updated[PHEROMONE_GRID][PHEROMONE_GRID];

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize pheromone grid
__global__ void initPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < PHEROMONE_GRID * PHEROMONE_GRID; i += stride) {
        int x = i % PHEROMONE_GRID;
        int y = i / PHEROMONE_GRID;
        d_pheromone[y][x] = 0.0f;
        d_phero_updated[y][x] = 0;
    }
}

// Decay pheromones each tick
__global__ void decayPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float decay = 0.95f; // 5% decay per tick
    
    for (int i = idx; i < PHEROMONE_GRID * PHEROMONE_GRID; i += stride) {
        int x = i % PHEROMONE_GRID;
        int y = i / PHEROMONE_GRID;
        d_pheromone[y][x] *= decay;
        if (d_pheromone[y][x] < 0.001f) d_pheromone[y][x] = 0.0f;
    }
}

// Initialize agents and resources
__global__ void init(Agent* agents, Resource* resources, int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS + RESOURCES) return;
    
    if (idx < AGENTS) {
        // Initialize agent
        Agent* a = &agents[idx];
        a->x = lcgf(&a->rng) * WORLD_SIZE;
        a->y = lcgf(&a->rng) * WORLD_SIZE;
        a->vx = (lcgf(&a->rng) - 0.5f) * 0.02f;
        a->vy = (lcgf(&a->rng) - 0.5f) * 0.02f;
        a->energy = 1.0f;
        a->fitness = 0.0f;
        a->rng = idx * 123456789 + 987654321;
        
        if (specialized) {
            // Specialized agents: one dominant role per archetype
            a->arch = idx % 4;
            for (int i = 0; i < 4; i++) a->role[i] = 0.1f;
            a->role[a->arch] = 0.7f;
        } else {
            // Uniform control: all roles equal
            a->arch = -1;
            for (int i = 0; i < 4; i++) a->role[i] = 0.25f;
        }
    } else {
        // Initialize resources
        int res_idx = idx - AGENTS;
        Resource* r = &resources[res_idx];
        r->rng = res_idx * 135791113 + 17192123;
        r->x = lcgf(&r->rng) * WORLD_SIZE;
        r->y = lcgf(&r->rng) * WORLD_SIZE;
        r->value = 0.8f + lcgf(&r->rng) * 0.4f;
        r->collected = 0;
    }
}

// Main simulation kernel
__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    if (a->energy <= 0.0f) return;
    
    // Anti-convergence: check similarity with random other agent
    int other_idx = (int)(lcgf(&a->rng) * AGENTS);
    if (other_idx >= AGENTS) other_idx = AGENTS - 1;
    Agent* other = &agents[other_idx];
    
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - other->role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Apply random drift to non-dominant role
        int drift_role = (int)(lcgf(&a->rng) * 4);
        while (drift_role == a->arch || a->arch == -1) {
            drift_role = (int)(lcgf(&a->rng) * 4);
        }
        a->role[drift_role] += (lcgf(&a->rng) - 0.5f) * 0.02f;
        
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < 4; i++) a->role[i] /= sum;
    }
    
    // Calculate behavior weights
    float explore_w = a->role[0];
    float collect_w = a->role[1];
    float comm_w = a->role[2];
    float defend_w = a->role[3];
    
    // Pheromone sensing (NOVEL MECHANISM)
    int cell_x = (int)(a->x / CELL_SIZE);
    int cell_y = (int)(a->y / CELL_SIZE);
    cell_x = max(0, min(PHEROMONE_GRID - 1, cell_x));
    cell_y = max(0, min(PHEROMONE_GRID - 1, cell_y));
    
    float phero_strength = d_pheromone[cell_y][cell_x];
    float phero_influence = 0.3f * phero_strength; // Max 30% influence
    
    // Sample nearby cells for gradient
    float grad_x = 0.0f, grad_y = 0.0f;
    if (phero_strength > 0.01f) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                if (nx >= 0 && nx < PHEROMONE_GRID && ny >= 0 && ny < PHEROMONE_GRID) {
                    float diff = d_pheromone[ny][nx] - phero_strength;
                    grad_x += diff * dx;
                    grad_y += diff * dy;
                }
            }
        }
        float grad_len = sqrtf(grad_x * grad_x + grad_y * grad_y);
        if (grad_len > 0.001f) {
            grad_x /= grad_len;
            grad_y /= grad_len;
        }
    }
    
    // Base movement from roles
    float move_x = a->vx;
    float move_y = a->vy;
    
    // Add pheromone gradient influence
    move_x += grad_x * phero_influence * explore_w;
    move_y += grad_y * phero_influence * explore_w;
    
    // Add random exploration
    move_x += (lcgf(&a->rng) - 0.5f) * 0.01f * explore_w;
    move_y += (lcgf(&a->rng) - 0.5f) * 0.01f * explore_w;
    
    // Normalize movement
    float move_len = sqrtf(move_x * move_x + move_y * move_y);
    if (move_len > 0.001f) {
        move_x /= move_len;
        move_y /= move_len;
    }
    
    // Apply movement
    a->x += move_x * 0.01f;
    a->y += move_y * 0.01f;
    
    // World boundaries
    if (a->x < 0.0f) { a->x = 0.0f; a->vx = -a->vx; }
    if (a->x > WORLD_SIZE) { a->x = WORLD_SIZE; a->vx = -a->vx; }
    if (a->y < 0.0f) { a->y = 0.0f; a->vy = -a->vy; }
    if (a->y > WORLD_SIZE) { a->y = WORLD_SIZE; a->vy = -a->vy; }
    
    // Resource interaction
    float best_dist = 100.0f;
    int best_res = -1;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx * dx + dy * dy);
        
        // Exploration detection range
        if (dist < 0.05f * explore_w && dist < best_dist) {
            best_dist = dist;
            best_res = i;
        }
        
        // Collection range with bonus
        if (dist < 0.03f * collect_w) {
            // Collect resource
            float bonus = (a->arch == ARCH_COLLECTOR) ? 1.5f : 1.0f;
            a->energy += r->value * bonus;
            a->fitness += r->value * bonus;
            r->collected = 1;
            
            // DEPOSIT PHEROMONE (NOVEL MECHANISM)
            int phero_x = (int)(r->x / CELL_SIZE);
            int phero_y = (int)(r->y / CELL_SIZE);
            if (phero_x >= 0 && phero_x < PHEROMONE_GRID && 
                phero_y >= 0 && phero_y < PHEROMONE_GRID) {
                atomicAdd(&d_pheromone[phero_y][phero_x], 0.5f);
            }
            break;
        }
    }
    
    // Communication behavior
    if (comm_w > 0.3f && best_res != -1) {
        Resource* r = &resources[best_res];
        // Broadcast to nearby agents
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* neighbor = &agents[i];
            float dx = neighbor->x - a->x;
            float dy = neighbor->y - a->y;
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist < 0.06f) {
                // Attract neighbor toward resource
                float influence = 0.02f * comm_w;
                neighbor->vx += (r->x - neighbor->x) * influence;
                neighbor->vy += (r->y - neighbor->y) * influence;
            }
        }
    }
    
    // Defense behavior: territory and perturbation resistance
    if (defend_w > 0.3f) {
        // Count nearby defenders of same archetype
        int defender_count = 0;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &agents[i];
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist < 0.04f && other->arch == ARCH_DEFENDER) {
                defender_count++;
            }
        }
        
        // Territory collection boost
        float boost = 1.0f + 0.2f * defender_count;
        a->energy *= boost;
        a->fitness *= boost;
        
        // Perturbation resistance (50% chance per 100 ticks)
        if (tick_num % 100 == 0 && lcgf(&a->rng) < 0.5f) {
            a->energy *= 0.5f; // Other agents lose half energy
        }
    }
    
    // Coupling with same archetype
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent* other = &agents[i];
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        float dist = sqrtf(dx * dx + dy * dy);
        
        if (dist < 0.02f) {
            float coupling = (a->arch == other->arch) ? 0.02f : 0.002f;
            for (int j = 0; j < 4; j++) {
                float diff = other->role[j] - a->role[j];
                a->role[j] += diff * coupling;
                other->role[j] -= diff * coupling;
            }
        }
    }
}

int main() {
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources_spec;
    Resource* d_resources_uniform;
    
    cudaMalloc(&d_agents_spec, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENTS * sizeof(Agent));
    cudaMalloc(&d_resources_spec, RESOURCES * sizeof(Resource));
    cudaMalloc(&d_resources_uniform, RESOURCES * sizeof(Resource));
    
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    
    // Initialize pheromone grid
    initPheromone<<<16, 256>>>();
    cudaDeviceSynchronize();
    
    // Initialize populations
    init<<<16, 256>>>(d_agents_spec, d_resources_spec, 1); // Specialized
    init<<<16, 256>>>(d_agents_uniform, d_resources_uniform, 0); // Uniform
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        // Decay pheromones
        decayPheromone<<<16, 256>>>();
        cudaDeviceSynchronize();
        
        // Run tick for both populations
        tick<<<16, 256>>>(d_agents_spec, d_resources_spec, t);
        tick<<<16, 256>>>(d_agents_uniform, d_resources_uniform, t);
        cudaDeviceSynchronize();
        
        // Respawn resources every 50 ticks
        if (t % 50 == 49) {
            init<<<1, 256>>>(d_agents_spec + AGENTS, d_resources_spec, 0);
            init<<<1, 256>>>(d_agents_uniform + AGENTS, d_resources_uniform, 0);
            cudaDeviceSynchronize();
        }
    }
    
    // Copy results back
    cudaMemcpy(h_agents_spec, d_agents_spec, AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_agents_uniform, d_agents_uniform, AGENTS * sizeof(Agent), cudaMemcpyDeviceTo