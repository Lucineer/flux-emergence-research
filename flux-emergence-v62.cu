
/*
CUDA Simulation Experiment v62: STIGMERGY TRAILS
Testing: Agents leave pheromone trails at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination beyond v8 baseline,
            especially for explore/communicate roles, leading to >1.61x ratio.
Baseline: v8 mechanisms (scarcity, territory, comms, anti-convergence, roles).
Novelty: Pheromone trails with spatial diffusion and decay.
Control: Uniform agents (all roles=0.25) vs Specialized (role[arch]=0.7).
Expected: Specialists should leverage trails more effectively, increasing ratio.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int BLOCK = 256;

// Pheromone grid constants
const int GRID_SIZE = 256;           // Spatial grid for pheromones
const float CELL_SIZE = 1.0f / GRID_SIZE;
const float PHEROMONE_DECAY = 0.95f; // Per tick decay
const float DIFFUSION_RATE = 0.1f;   // Spread to neighbors
const float DEPOSIT_STRENGTH = 0.5f; // When resource collected

// Agent archetype
enum Archetype {
    ARCH_EXPLORER = 0,
    ARCH_COLLECTOR,
    ARCH_COMMUNICATOR,
    ARCH_DEFENDER,
    ARCH_COUNT
};

// Agent structure
struct Agent {
    float x, y;           // Position
    float vx, vy;         // Velocity
    float energy;         // Energy level
    float role[4];        // Role strengths: explore, collect, communicate, defend
    float fitness;        // Performance metric
    int arch;             // Archetype (0-3)
    unsigned int rng;     // RNG state
};

// Resource structure
struct Resource {
    float x, y;           // Position
    float value;          // Resource value
    int collected;        // Whether collected
};

// Pheromone grid (global memory)
__device__ float pheromoneGrid[GRID_SIZE][GRID_SIZE];

// Linear congruential generator
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Initialize pheromone grid to zero
__global__ void initPheromones() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < GRID_SIZE * GRID_SIZE; i += stride) {
        int x = i % GRID_SIZE;
        int y = i / GRID_SIZE;
        pheromoneGrid[y][x] = 0.0f;
    }
}

// Diffuse and decay pheromones
__global__ void updatePheromones() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // First, diffuse in shared memory
    __shared__ float tempGrid[GRID_SIZE][GRID_SIZE];
    
    for (int i = idx; i < GRID_SIZE * GRID_SIZE; i += stride) {
        int x = i % GRID_SIZE;
        int y = i / GRID_SIZE;
        tempGrid[y][x] = pheromoneGrid[y][x];
    }
    __syncthreads();
    
    for (int i = idx; i < GRID_SIZE * GRID_SIZE; i += stride) {
        int x = i % GRID_SIZE;
        int y = i / GRID_SIZE;
        
        // Diffusion (simple 4-neighbor)
        float sum = tempGrid[y][x] * (1.0f - DIFFUSION_RATE);
        int count = 1;
        
        if (x > 0) { sum += tempGrid[y][x-1] * DIFFUSION_RATE * 0.25f; count++; }
        if (x < GRID_SIZE-1) { sum += tempGrid[y][x+1] * DIFFUSION_RATE * 0.25f; count++; }
        if (y > 0) { sum += tempGrid[y-1][x] * DIFFUSION_RATE * 0.25f; count++; }
        if (y < GRID_SIZE-1) { sum += tempGrid[y+1][x] * DIFFUSION_RATE * 0.25f; count++; }
        
        // Decay
        pheromoneGrid[y][x] = sum * PHEROMONE_DECAY / (1.0f + (count-1)*0.25f);
    }
}

// Deposit pheromone at location
__device__ void depositPheromone(float x, float y, float amount) {
    int gridX = min(GRID_SIZE-1, max(0, (int)(x / CELL_SIZE)));
    int gridY = min(GRID_SIZE-1, max(0, (int)(y / CELL_SIZE)));
    
    atomicAdd(&pheromoneGrid[gridY][gridX], amount);
}

// Sample pheromone at location
__device__ float samplePheromone(float x, float y) {
    int gridX = min(GRID_SIZE-1, max(0, (int)(x / CELL_SIZE)));
    int gridY = min(GRID_SIZE-1, max(0, (int)(y / CELL_SIZE)));
    
    return pheromoneGrid[gridY][gridX];
}

// Initialize agents
__global__ void initAgents(Agent* agents, int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
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
        // Specialized population: each agent has one strong role
        a->arch = idx % ARCH_COUNT;
        for (int i = 0; i < 4; i++) {
            a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform control: all roles equal
        a->arch = ARCH_EXPLORER;
        for (int i = 0; i < 4; i++) {
            a->role[i] = 0.25f;
        }
    }
}

// Initialize resources
__global__ void initResources(Resource* resources) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    Resource* r = &resources[idx];
    unsigned int seed = idx * 19 + 54321;
    r->x = (seed * 1103515245 + 12345) / 4294967296.0f;
    r->y = ((seed * 1103515245 + 12345) * 1103515245 + 12345) / 4294967296.0f;
    r->value = 0.5f + (seed % 100) / 200.0f; // 0.5-1.0
    r->collected = 0;
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, int tickNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: check similarity with random other agent
    int otherIdx = (idx + 37) % AGENTS;
    Agent* other = &agents[otherIdx];
    
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - other->role[i]);
    }
    similarity = 1.0f - similarity / 4.0f;
    
    if (similarity > 0.9f) {
        // Random drift on non-dominant role
        int driftRole = (int)(lcgf(&a->rng) * 4);
        while (driftRole == a->arch) {
            driftRole = (int)(lcgf(&a->rng) * 4);
        }
        a->role[driftRole] += (lcgf(&a->rng) * 0.02f - 0.01f);
        a->role[driftRole] = max(0.0f, min(1.0f, a->role[driftRole]));
    }
    
    // Coupling: adjust toward same archetype, away from different
    float couplingSame = 0.02f;
    float couplingDiff = 0.002f;
    
    for (int i = 0; i < 4; i++) {
        if (i == a->arch) {
            a->role[i] += (other->role[i] - a->role[i]) * couplingSame;
        } else {
            a->role[i] += (other->role[i] - a->role[i]) * couplingDiff;
        }
        a->role[i] = max(0.0f, min(1.0f, a->role[i]));
    }
    
    // Normalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) {
        a->role[i] /= sum;
    }
    
    // Movement with pheromone influence
    float exploreStrength = a->role[0];
    float defendStrength = a->role[3];
    
    // Sample pheromone gradient
    float px = a->x;
    float py = a->y;
    float phere = samplePheromone(px, py);
    float phereRight = samplePheromone(min(1.0f, px + 0.01f), py);
    float phereUp = samplePheromone(px, min(1.0f, py + 0.01f));
    
    // Move toward higher pheromone (weighted by explore role)
    a->vx += (phereRight - phere) * exploreStrength * 0.5f;
    a->vy += (phereUp - phere) * exploreStrength * 0.5f;
    
    // Add some random motion
    a->vx += lcgf(&a->rng) * 0.004f - 0.002f;
    a->vy += lcgf(&a->rng) * 0.004f - 0.002f;
    
    // Velocity limits
    float speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
    if (speed > 0.03f) {
        a->vx *= 0.03f / speed;
        a->vy *= 0.03f / speed;
    }
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary wrap
    if (a->x < 0) a->x = 1.0f + a->x;
    if (a->x > 1) a->x = a->x - 1.0f;
    if (a->y < 0) a->y = 1.0f + a->y;
    if (a->y > 1) a->y = a->y - 1.0f;
    
    // Resource interaction
    float collectRange = 0.02f + a->role[1] * 0.02f; // 0.02-0.04
    float detectRange = 0.03f + a->role[0] * 0.04f;  // 0.03-0.07
    
    int nearestRes = -1;
    float nearestDist = 1e6;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Wrap-around distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx * dx + dy * dy);
        
        if (dist < detectRange && dist < nearestDist) {
            nearestDist = dist;
            nearestRes = i;
        }
        
        // Collect if in range
        if (dist < collectRange) {
            float bonus = 1.0f + a->role[1] * 0.5f; // Up to 50% bonus
            a->energy += r->value * bonus;
            a->fitness += r->value * bonus;
            r->collected = 1;
            
            // DEPOSIT PHEROMONE at resource location (novel mechanism)
            depositPheromone(r->x, r->y, DEPOSIT_STRENGTH * (1.0f + a->role[2]));
        }
    }
    
    // Communication (broadcast nearest resource)
    if (a->role[2] > 0.3f && nearestRes >= 0) {
        float commRange = 0.06f;
        Resource* r = &resources[nearestRes];
        
        // In real implementation, would broadcast to neighbors
        // For simplicity, deposit pheromone at resource location
        depositPheromone(r->x, r->y, DEPOSIT_STRENGTH * a->role[2] * 0.5f);
    }
    
    // Territory defense boost
    float territoryBoost = 1.0f;
    int defenderCount = 0;
    
    // Check for nearby defenders (simplified)
    for (int i = 0; i < min(10, AGENTS); i++) {
        int checkIdx = (idx + i * 37) % AGENTS;
        if (checkIdx == idx) continue;
        
        Agent* other = &agents[checkIdx];
        if (other->arch == ARCH_DEFENDER) {
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            if (dx > 0.5f) dx -= 1.0f;
            if (dx < -0.5f) dx += 1.0f;
            if (dy > 0.5f) dy -= 1.0f;
            if (dy < -0.5f) dy += 1.0f;
            
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < 0.1f) {
                defenderCount++;
            }
        }
    }
    
    territoryBoost += defenderCount * 0.2f; // 20% per defender
    a->energy *= territoryBoost;
    
    // Perturbation (every 50 ticks)
    if (tickNum % 50 == 0) {
        float resistance = a->role[3]; // Defend role provides resistance
        if (lcgf(&a->rng) > resistance * 0.5f) {
            a->energy *= 0.5f; // Halve energy
        }
    }
}

// Reset resources periodically
__global__ void resetResources(Resource* resources, int tickNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    // Respawn every 50 ticks
    if (tickNum % 50 == 0) {
        Resource* r = &resources[idx];
        unsigned int seed = idx * 19 + 54321 + tickNum;
        r->x = (seed * 1103515245 + 12345) / 4294967296.0f;
        r->y = ((seed * 1103515245 + 12345) * 1103515245 + 12345) / 4294967296.0f;
        r->collected = 0;
    }
}

int main() {
    printf("Experiment v62: STIGMERGY TRAILS\n");
    printf("Testing pheromone trails with diffusion/decay\n");
    printf("Prediction: Specialists >1.61x vs uniform\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RESOURCES, TICKS);
    
    // Allocate device memory
    Agent* d_agentsA;
    Agent* d_agentsB;
    Resource* d_resources;
    
    cudaMalloc(&d_agentsA, AGENTS * sizeof(Agent));
    cudaMalloc(&d_agentsB, AGENTS * sizeof(Agent));
    cudaMalloc(&