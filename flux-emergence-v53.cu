
/*
CUDA Simulation Experiment v53: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination beyond baseline v8,
            especially for explore/collect roles, leading to >1.61x advantage.
Baseline: v8 mechanisms (scarcity, territory, comms) included.
Novelty: Pheromone trails that persist for 50 ticks, guiding agents to resources.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define AGENTS 1024
#define RESOURCES 128
#define TICKS 500
#define PHEROMONE_GRID_SIZE 256
#define PHEROMONE_DECAY 0.98f
#define PHEROMONE_DURATION 50

struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];  // 0:explore, 1:collect, 2:communicate, 3:defend
    float fitness;
    int arch;       // 0:uniform, 1:specialized
    unsigned int rng;
};

struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone grid stored in global memory
__device__ float pheromone[PHEROMONE_GRID_SIZE][PHEROMONE_GRID_SIZE];
__device__ int pheromone_age[PHEROMONE_GRID_SIZE][PHEROMONE_GRID_SIZE];

// LCG RNG functions
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
    
    for (int i = idx; i < PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE; i += stride) {
        int x = i % PHEROMONE_GRID_SIZE;
        int y = i / PHEROMONE_GRID_SIZE;
        pheromone[x][y] = 0.0f;
        pheromone_age[x][y] = 0;
    }
}

// Decay pheromones
__global__ void decayPheromone() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE; i += stride) {
        int x = i % PHEROMONE_GRID_SIZE;
        int y = i / PHEROMONE_GRID_SIZE;
        
        if (pheromone_age[x][y] > 0) {
            pheromone[x][y] *= PHEROMONE_DECAY;
            pheromone_age[x][y]--;
            if (pheromone_age[x][y] == 0) {
                pheromone[x][y] = 0.0f;
            }
        }
    }
}

// Add pheromone at location
__device__ void addPheromone(float x, float y, float strength) {
    int gx = min(max((int)(x * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int gy = min(max((int)(y * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    
    atomicAdd(&pheromone[gx][gy], strength);
    pheromone_age[gx][gy] = PHEROMONE_DURATION;
}

// Sample pheromone gradient
__device__ void samplePheromone(float x, float y, float* grad_x, float* grad_y) {
    int gx = min(max((int)(x * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    int gy = min(max((int)(y * PHEROMONE_GRID_SIZE), 0), PHEROMONE_GRID_SIZE - 1);
    
    float center = pheromone[gx][gy];
    float left = (gx > 0) ? pheromone[gx-1][gy] : 0.0f;
    float right = (gx < PHEROMONE_GRID_SIZE-1) ? pheromone[gx+1][gy] : 0.0f;
    float up = (gy > 0) ? pheromone[gx][gy-1] : 0.0f;
    float down = (gy < PHEROMONE_GRID_SIZE-1) ? pheromone[gx][gy+1] : 0.0f;
    
    *grad_x = (right - left) * 0.5f;
    *grad_y = (down - up) * 0.5f;
}

__global__ void initAgents(Agent* agents, Resource* resources) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS + RESOURCES) return;
    
    if (idx < AGENTS) {
        Agent* a = &agents[idx];
        a->x = lcgf(&a->rng);
        a->y = lcgf(&a->rng);
        a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
        a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
        a->energy = 1.0f;
        a->fitness = 0.0f;
        a->rng = idx * 123456789 + 987654321;
        
        // Assign architecture: half uniform, half specialized
        a->arch = (idx < AGENTS/2) ? 0 : 1;
        
        if (a->arch == 0) {  // Uniform control
            a->role[0] = 0.25f; a->role[1] = 0.25f;
            a->role[2] = 0.25f; a->role[3] = 0.25f;
        } else {  // Specialized
            int primary = idx % 4;  // Each specialist excels at one role
            for (int i = 0; i < 4; i++) {
                a->role[i] = (i == primary) ? 0.7f : 0.1f;
            }
        }
    } else {
        int ridx = idx - AGENTS;
        Resource* r = &resources[ridx];
        r->x = lcgf(&agents[0].rng);
        r->y = lcgf(&agents[0].rng);
        r->value = 0.8f + lcgf(&agents[0].rng) * 0.4f;
        r->collected = 0;
    }
}

__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect similarity with random neighbor
    int neighbor = lcg(&a->rng) % AGENTS;
    float similarity = 0.0f;
    for (int i = 0; i < 4; i++) {
        similarity += fabsf(a->role[i] - agents[neighbor].role[i]);
    }
    similarity = 1.0f - similarity * 0.25f;
    
    if (similarity > 0.9f) {
        // Find non-dominant role
        float max_role = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < 4; i++) {
            if (a->role[i] > max_role) {
                max_role = a->role[i];
                max_idx = i;
            }
        }
        int drift_idx = (max_idx + 1 + (lcg(&a->rng) % 3)) % 4;
        a->role[drift_idx] += (lcgf(&a->rng) * 0.02f - 0.01f);
        a->role[drift_idx] = fmaxf(0.0f, fminf(1.0f, a->role[drift_idx]));
    }
    
    // Normalize roles
    float sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int i = 0; i < 4; i++) a->role[i] /= sum;
    
    // Movement with pheromone guidance
    float grad_x, grad_y;
    samplePheromone(a->x, a->y, &grad_x, &grad_y);
    
    // Explore role influences random movement + pheromone following
    float explore_strength = a->role[0] * 0.1f;
    a->vx += (lcgf(&a->rng) * 0.02f - 0.01f) + grad_x * explore_strength;
    a->vy += (lcgf(&a->rng) * 0.02f - 0.01f) + grad_y * explore_strength;
    
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
    if (a->x < 0.0f) a->x = 1.0f + a->x;
    if (a->x > 1.0f) a->x = a->x - 1.0f;
    if (a->y < 0.0f) a->y = 1.0f + a->y;
    if (a->y > 1.0f) a->y = a->y - 1.0f;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_idx = -1;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        // Boundary wrap distance
        if (dx > 0.5f) dx -= 1.0f;
        if (dx < -0.5f) dx += 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        if (dy < -0.5f) dy += 1.0f;
        
        float dist = sqrtf(dx * dx + dy * dy);
        
        // Detection range based on explore role
        float detect_range = 0.03f + a->role[0] * 0.04f;
        if (dist < detect_range && dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
        
        // Collection range based on collect role
        float grab_range = 0.02f + a->role[1] * 0.02f;
        if (dist < grab_range) {
            // Collectors get bonus
            float bonus = 1.0f + a->role[1] * 0.5f;
            a->energy += r->value * bonus;
            a->fitness += r->value * bonus;
            
            // Leave pheromone at resource location
            addPheromone(r->x, r->y, 1.0f + a->role[1]);
            
            r->collected = 1;
            best_idx = -1;
            break;
        }
    }
    
    // Communication role: broadcast found resources
    if (a->role[2] > 0.3f && best_idx != -1) {
        Resource* r = &resources[best_idx];
        for (int i = 0; i < 4; i++) {
            int other = lcg(&a->rng) % AGENTS;
            // Only communicate with same archetype (coupling)
            if (agents[other].arch == a->arch) {
                float dx = r->x - agents[other].x;
                float dy = r->y - agents[other].y;
                if (dx > 0.5f) dx -= 1.0f;
                if (dx < -0.5f) dx += 1.0f;
                if (dy > 0.5f) dy -= 1.0f;
                if (dy < -0.5f) dy += 1.0f;
                
                if (fabsf(dx) < 0.06f && fabsf(dy) < 0.06f) {
                    agents[other].vx += dx * 0.01f * a->role[2];
                    agents[other].vy += dy * 0.01f * a->role[2];
                }
            }
        }
    }
    
    // Defense role: territory and perturbation resistance
    if (a->role[3] > 0.3f) {
        // Count nearby defenders of same archetype
        int defender_count = 0;
        for (int i = 0; i < 16; i++) {
            int other = lcg(&a->rng) % AGENTS;
            if (agents[other].arch == a->arch && agents[other].role[3] > 0.3f) {
                float dx = agents[other].x - a->x;
                float dy = agents[other].y - a->y;
                if (dx > 0.5f) dx -= 1.0f;
                if (dx < -0.5f) dx += 1.0f;
                if (dy > 0.5f) dy -= 1.0f;
                if (dy < -0.5f) dy += 1.0f;
                
                if (fabsf(dx) < 0.05f && fabsf(dy) < 0.05f) {
                    defender_count++;
                }
            }
        }
        
        // Territory bonus
        float territory_bonus = 1.0f + defender_count * 0.2f * a->role[3];
        a->energy *= territory_bonus;
        a->fitness *= territory_bonus;
        
        // Perturbation resistance
        if (tick_num % 100 == 0 && lcgf(&a->rng) < 0.1f) {
            a->energy *= (0.5f + a->role[3] * 0.5f);
        }
    }
    
    // Energy limits
    if (a->energy > 2.0f) a->energy = 2.0f;
    if (a->energy < 0.0f) a->energy = 0.0f;
}

int main() {
    printf("Experiment v53: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone markers enhance specialist coordination\n");
    printf("Prediction: Specialist advantage >1.61x (baseline v8)\n");
    printf("Agents: %d (half uniform, half specialized)\n", AGENTS);
    printf("Resources: %d (scarce)\n", RESOURCES);
    printf("Ticks: %d\n", TICKS);
    printf("Pheromone grid: %dx%d, decay=%.2f, duration=%d ticks\n\n",
           PHEROMONE_GRID_SIZE, PHEROMONE_GRID_SIZE, PHEROMONE_DECAY, PHEROMONE_DURATION);
    
    // Allocate memory
    Agent* agents;
    Resource* resources;
    cudaMallocManaged(&agents, AGENTS * sizeof(Agent));
    cudaMallocManaged(&resources, RESOURCES * sizeof(Resource));
    
    // Initialize
    initPheromone<<<16, 256>>>();
    cudaDeviceSynchronize();
    
    initAgents<<<16, 256>>>(agents, resources);
    cudaDeviceSynchronize();
    
    // Main simulation loop
    for (int t = 0; t < TICKS; t++) {
        // Decay pheromones every tick
        decayPheromone<<<16, 256>>>();
        cudaDeviceSynchronize();
        
        // Run agent tick
        tick<<<16, 256>>>(agents, resources, t);
        cudaDeviceSynchronize();
        
        // Respawn resources periodically
        if (t % 50 == 0) {
            for (int i = 0; i < RESOURCES; i++) {
                if (resources[i].collected || lcgf(&agents[0].rng) < 0.3f) {
                    resources[i].x = lcgf(&agents[0].rng);
                    resources[i].y = lcgf(&agents[0].rng);
                    resources[i].value = 0.8f