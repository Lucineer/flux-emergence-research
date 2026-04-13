// CUDA Simulation Experiment v58: STIGMERGY TRAILS
// Testing: Pheromone trails at resource locations that decay over time
// Prediction: Stigmergy will amplify specialist advantage by 2.0x+ (vs v8's 1.61x)
// Mechanism: Agents deposit pheromone when collecting resources, others follow trails
// Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK_SIZE = 256;

// Agent archetypes
enum { ARCH_EXPLORER = 0, ARCH_COLLECTOR = 1, ARCH_COMMUNICATOR = 2, ARCH_DEFENDER = 3, ARCH_COUNT = 4 };

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Behavioral roles: explore, collect, communicate, defend
    float fitness;        // Fitness score
    int arch;             // Dominant archetype
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Collection flag
    float pheromone;      // NEW: Pheromone trail strength at this location
};

// Stigmergy parameters
const float PHEROMONE_DEPOSIT = 0.8f;    // Amount deposited per collection
const float PHEROMONE_DECAY = 0.95f;     // Decay per tick
const float PHEROMONE_THRESHOLD = 0.1f;  // Minimum to follow
const float TRAIL_FOLLOW_STRENGTH = 0.3f;// How strongly trails influence movement

// v8 baseline parameters
const float COUPLING_SAME = 0.02f;
const float COUPLING_DIFF = 0.002f;
const float ANTICONVERGENCE_THRESHOLD = 0.9f;
const float DRIFT_STRENGTH = 0.01f;
const float ENERGY_DECAY = 0.999f;
const float PERTURBATION_ENERGY_LOSS = 0.5f;

// Role-specific parameters
const float EXPLORE_RANGE_MIN = 0.03f;
const float EXPLORE_RANGE_MAX = 0.07f;
const float COLLECT_RANGE = 0.03f;
const float COLLECT_BONUS = 1.5f;
const float COMM_RANGE = 0.06f;
const float DEFEND_BOOST_PER_NEIGHBOR = 0.2f;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize agents
__global__ void init_agents(Agent* agents, int specialized) {
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
    
    if (specialized) {
        // Specialized population: each agent strong in one role
        a->arch = idx % ARCH_COUNT;
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles equal
        a->arch = ARCH_EXPLORER;
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void init_resources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = idx * 19 + 54321;
    resources[idx].x = lcgf(&rng);
    resources[idx].y = lcgf(&rng);
    resources[idx].value = 0.5f + lcgf(&rng) * 0.5f;
    resources[idx].collected = 0;
    resources[idx].pheromone = 0.0f;  // Start with no pheromone
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Apply energy decay
    a->energy *= ENERGY_DECAY;
    
    // Apply anti-convergence drift
    float max_role = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < ARCH_COUNT; i++) {
        if (a->role[i] > max_role) {
            max_role = a->role[i];
            max_idx = i;
        }
    }
    
    if (max_role > ANTICONVERGENCE_THRESHOLD) {
        // Apply random drift to non-dominant roles
        for (int i = 0; i < ARCH_COUNT; i++) {
            if (i != max_idx) {
                float drift = (lcgf(&a->rng) - 0.5f) * 2.0f * DRIFT_STRENGTH;
                a->role[i] += drift;
                a->role[i] = fmaxf(0.0f, fminf(1.0f, a->role[i]));
            }
        }
        // Renormalize
        float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
        for (int i = 0; i < ARCH_COUNT; i++) {
            a->role[i] /= sum;
        }
    }
    
    // Find nearest resource with pheromone trail
    float best_pheromone = 0.0f;
    float target_x = 0.0f, target_y = 0.0f;
    int found_trail = 0;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist2 = dx * dx + dy * dy;
        
        // Check if within detection range (explore role)
        float explore_range = EXPLORE_RANGE_MIN + 
                            (EXPLORE_RANGE_MAX - EXPLORE_RANGE_MIN) * a->role[ARCH_EXPLORER];
        
        if (dist2 < explore_range * explore_range) {
            // NEW: Prefer resources with stronger pheromone trails
            if (r->pheromone > best_pheromone && r->pheromone > PHEROMONE_THRESHOLD) {
                best_pheromone = r->pheromone;
                target_x = r->x;
                target_y = r->y;
                found_trail = 1;
            }
        }
    }
    
    // Movement with stigmergy influence
    if (found_trail) {
        // Follow pheromone trail
        float dx = target_x - a->x;
        float dy = target_y - a->y;
        float dist = sqrtf(dx * dx + dy * dy + 1e-6f);
        a->vx += (dx / dist) * TRAIL_FOLLOW_STRENGTH * a->role[ARCH_EXPLORER];
        a->vy += (dy / dist) * TRAIL_FOLLOW_STRENGTH * a->role[ARCH_EXPLORER];
    } else {
        // Random exploration
        a->vx += (lcgf(&a->rng) - 0.5f) * 0.01f;
        a->vy += (lcgf(&a->rng) - 0.5f) * 0.01f;
    }
    
    // Velocity damping
    a->vx *= 0.98f;
    a->vy *= 0.98f;
    
    // Update position (bounded)
    a->x += a->vx;
    a->y += a->vy;
    if (a->x < 0.0f) { a->x = 0.0f; a->vx = -a->vx * 0.5f; }
    if (a->x > 1.0f) { a->x = 1.0f; a->vx = -a->vx * 0.5f; }
    if (a->y < 0.0f) { a->y = 0.0f; a->vy = -a->vy * 0.5f; }
    if (a->y > 1.0f) { a->y = 1.0f; a->vy = -a->vy * 0.5f; }
    
    // Resource collection
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < COLLECT_RANGE * COLLECT_RANGE) {
            // Collector bonus
            float gain = r->value * (1.0f + (COLLECT_BONUS - 1.0f) * a->role[ARCH_COLLECTOR]);
            
            // Defender territory boost
            int defender_count = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent* other = &agents[j];
                if (other->arch == ARCH_DEFENDER) {
                    float odx = other->x - a->x;
                    float ody = other->y - a->y;
                    if (odx * odx + ody * ody < 0.05f * 0.05f) {
                        defender_count++;
                    }
                }
            }
            gain *= (1.0f + DEFEND_BOOST_PER_NEIGHBOR * defender_count);
            
            a->energy += gain;
            a->fitness += gain;
            r->collected = 1;
            
            // NEW: Deposit pheromone at collection site
            r->pheromone += PHEROMONE_DEPOSIT * (1.0f + a->role[ARCH_COLLECTOR]);
            break;
        }
    }
    
    // Communication (broadcast nearest resource)
    if (a->role[ARCH_COMMUNICATOR] > 0.3f) {
        float nearest_dist = 1e6f;
        float nx = 0.0f, ny = 0.0f;
        int found = 0;
        
        for (int i = 0; i < RESOURCES; i++) {
            Resource* r = &resources[i];
            if (r->collected) continue;
            
            float dx = r->x - a->x;
            float dy = r->y - a->y;
            float dist2 = dx * dx + dy * dy;
            
            if (dist2 < nearest_dist) {
                nearest_dist = dist2;
                nx = r->x;
                ny = r->y;
                found = 1;
            }
        }
        
        if (found) {
            // Broadcast to nearby agents
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent* other = &agents[j];
                
                float dx = other->x - a->x;
                float dy = other->y - a->y;
                if (dx * dx + dy * dy < COMM_RANGE * COMM_RANGE) {
                    // Influence receiver's movement toward resource
                    float influence = a->role[ARCH_COMMUNICATOR] * 0.1f;
                    float rdx = nx - other->x;
                    float rdy = ny - other->y;
                    float rdist = sqrtf(rdx * rdx + rdy * rdy + 1e-6f);
                    other->vx += (rdx / rdist) * influence;
                    other->vy += (rdy / rdist) * influence;
                }
            }
        }
    }
    
    // Social coupling
    for (int j = 0; j < AGENTS; j++) {
        if (j == idx) continue;
        Agent* other = &agents[j];
        
        float dx = other->x - a->x;
        float dy = other->y - a->y;
        float dist2 = dx * dx + dy * dy;
        
        if (dist2 < 0.05f * 0.05f) {
            float coupling = (a->arch == other->arch) ? COUPLING_SAME : COUPLING_DIFF;
            
            for (int k = 0; k < ARCH_COUNT; k++) {
                float diff = other->role[k] - a->role[k];
                a->role[k] += diff * coupling;
            }
        }
    }
    
    // Renormalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < ARCH_COUNT; i++) {
        a->role[i] /= sum;
    }
    
    // Periodic perturbation (every 50 ticks)
    if (tick_num % 50 == 25) {
        // Defenders resist perturbation
        float resistance = a->role[ARCH_DEFENDER];
        if (lcgf(&a->rng) > resistance * 0.5f) {
            a->energy *= PERTURBATION_ENERGY_LOSS;
            // Random displacement
            a->x += (lcgf(&a->rng) - 0.5f) * 0.2f;
            a->y += (lcgf(&a->rng) - 0.5f) * 0.2f;
            a->x = fmaxf(0.0f, fminf(1.0f, a->x));
            a->y = fmaxf(0.0f, fminf(1.0f, a->y));
        }
    }
}

// Update pheromone trails (decay and respawn resources)
__global__ void update_stigmergy(Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    
    // Decay pheromone
    r->pheromone *= PHEROMONE_DECAY;
    
    // Respawn resources every 50 ticks
    if (tick_num % 50 == 0) {
        if (r->collected) {
            r->collected = 0;
            unsigned int rng = idx * 19 + tick_num * 7 + 54321;
            r->x = lcgf(&rng);
            r->y = lcgf(&rng);
            r->value = 0.5f + lcgf(&rng) * 0.5f;
            r->pheromone = 0.0f;  // Reset pheromone on respawn
        }
    }
}

int main() {
    printf("=== CUDA Simulation Experiment v58: STIGMERGY TRAILS ===\n");
    printf("Testing: Pheromone trails at resource locations\n");
    printf("Prediction: Stigmergy amplifies specialist advantage to 2.0x+\n");
    printf("Baseline: v8 mechanisms (scarcity, territory, comms, anti-convergence)\n\n");
    
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources_spec;
    Resource* d_resources_uniform;
    
    Agent* h_agents_spec = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    
    cudaMalloc(&d_agents_spec, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_agents_uniform, sizeof(Agent) * AGENTS);
    cudaMalloc(&d_resources_spec, sizeof(Resource) * RESOURCES);
    cudaMalloc(&d_resources_uniform, sizeof(Resource) * RESOURCES);
    
    // Initialize
    int blocks = (AGENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int res_blocks = (RESOURCES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    init_agents<<<blocks, BLOCK_SIZE>>>(d_agents_spec, 1);
