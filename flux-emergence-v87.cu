
/*
CUDA Simulation Experiment v87: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will improve specialist efficiency by 20-30% over baseline v8,
            as they create persistent environmental memory that guides exploration.
Baseline: v8 mechanisms (scarcity, territory, communication, anti-convergence) included.
Novelty: Pheromone trails with spatial diffusion and decay.
Comparison: Specialized archetypes (role[arch]=0.7) vs uniform control (all roles=0.25).
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENT_COUNT = 1024;
const int RESOURCE_COUNT = 128;
const int TICKS = 500;
const int PHEROMONE_GRID_SIZE = 256;
const float WORLD_SIZE = 1.0f;
const float PHEROMONE_DECAY = 0.97f;
const float DIFFUSION_RATE = 0.1f;
const float PHEROMONE_STRENGTH = 0.5f;

// Agent archetypes
enum Archetype { EXPLORER, COLLECTOR, COMMUNICATOR, DEFENDER, ARCHETYPE_COUNT };

// Agent structure
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPE_COUNT];
    float fitness;
    int arch;
    unsigned int rng;
};

// Resource structure
struct Resource {
    float x, y;
    float value;
    int collected;
};

// Pheromone grid (device-only)
__device__ float d_pheromone[PHEROMONE_GRID_SIZE][PHEROMONE_GRID_SIZE];

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
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < PHEROMONE_GRID_SIZE && idy < PHEROMONE_GRID_SIZE) {
        d_pheromone[idx][idy] = 0.0f;
    }
}

// Diffuse and decay pheromones
__global__ void updatePheromone() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= 1 && x < PHEROMONE_GRID_SIZE-1 && y >= 1 && y < PHEROMONE_GRID_SIZE-1) {
        float sum = d_pheromone[x][y] * (1.0f - DIFFUSION_RATE * 4);
        sum += DIFFUSION_RATE * (d_pheromone[x-1][y] + d_pheromone[x+1][y] +
                                 d_pheromone[x][y-1] + d_pheromone[x][y+1]);
        d_pheromone[x][y] = fmaxf(0.0f, sum * PHEROMONE_DECAY);
    }
}

// Add pheromone at location
__device__ void addPheromone(float wx, float wy, float amount) {
    int gx = (int)((wx / WORLD_SIZE) * (PHEROMONE_GRID_SIZE-1));
    int gy = (int)((wy / WORLD_SIZE) * (PHEROMONE_GRID_SIZE-1));
    gx = max(0, min(PHEROMONE_GRID_SIZE-1, gx));
    gy = max(0, min(PHEROMONE_GRID_SIZE-1, gy));
    
    atomicAdd(&d_pheromone[gx][gy], amount);
}

// Sample pheromone at location
__device__ float samplePheromone(float wx, float wy) {
    float fx = (wx / WORLD_SIZE) * (PHEROMONE_GRID_SIZE-1);
    float fy = (wy / WORLD_SIZE) * (PHEROMONE_GRID_SIZE-1);
    
    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = min(x0+1, PHEROMONE_GRID_SIZE-1);
    int y1 = min(y0+1, PHEROMONE_GRID_SIZE-1);
    
    float tx = fx - x0;
    float ty = fy - y0;
    
    return (1-tx)*(1-ty)*d_pheromone[x0][y0] +
           tx*(1-ty)*d_pheromone[x1][y0] +
           (1-tx)*ty*d_pheromone[x0][y1] +
           tx*ty*d_pheromone[x1][y1];
}

// Initialize agents and resources
__global__ void init(Agent* agents, Resource* resources, int specialized) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize agents
    if (idx < AGENT_COUNT) {
        Agent* a = &agents[idx];
        a->rng = idx * 12345 + 1;
        a->x = lcgf(&a->rng);
        a->y = lcgf(&a->rng);
        a->vx = lcgf(&a->rng) * 0.02f - 0.01f;
        a->vy = lcgf(&a->rng) * 0.02f - 0.01f;
        a->energy = 1.0f;
        a->fitness = 0.0f;
        
        if (specialized) {
            // Specialized: one dominant role
            a->arch = idx % ARCHETYPE_COUNT;
            for (int i = 0; i < ARCHETYPE_COUNT; i++) {
                a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
            }
        } else {
            // Uniform control
            a->arch = -1;
            for (int i = 0; i < ARCHETYPE_COUNT; i++) {
                a->role[i] = 0.25f;
            }
        }
    }
    
    // Initialize resources
    if (idx < RESOURCE_COUNT) {
        Resource* r = &resources[idx];
        unsigned int rng = idx * 67890 + 1;
        r->x = lcgf(&rng);
        r->y = lcgf(&rng);
        r->value = 0.5f + lcgf(&rng) * 0.5f;
        r->collected = 0;
    }
}

// Main simulation tick
__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENT_COUNT) return;
    
    Agent* a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: detect similarity with random agent
    int other_idx = (int)(lcgf(&a->rng) * AGENT_COUNT);
    if (other_idx >= AGENT_COUNT) other_idx = AGENT_COUNT - 1;
    Agent* other = &agents[other_idx];
    
    float similarity = 0.0f;
    for (int i = 0; i < ARCHETYPE_COUNT; i++) {
        similarity += fabsf(a->role[i] - other->role[i]);
    }
    similarity = 1.0f - similarity / ARCHETYPE_COUNT;
    
    if (similarity > 0.9f) {
        // Apply drift to non-dominant role
        int drift_role = (int)(lcgf(&a->rng) * ARCHETYPE_COUNT);
        if (a->arch >= 0 && drift_role == a->arch) {
            drift_role = (drift_role + 1) % ARCHETYPE_COUNT;
        }
        a->role[drift_role] += lcgf(&a->rng) * 0.02f - 0.01f;
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < ARCHETYPE_COUNT; i++) sum += a->role[i];
        for (int i = 0; i < ARCHETYPE_COUNT; i++) a->role[i] /= sum;
    }
    
    // Pheromone-guided movement (novel mechanism)
    float pheromone = samplePheromone(a->x, a->y);
    float explore_bias = a->role[EXPLORER] * 0.1f;
    
    // Move toward higher pheromone concentrations with explore probability
    float dx = 0.0f, dy = 0.0f;
    if (lcgf(&a->rng) < explore_bias) {
        // Random exploration
        dx = lcgf(&a->rng) * 0.02f - 0.01f;
        dy = lcgf(&a->rng) * 0.02f - 0.01f;
    } else {
        // Pheromone gradient following
        float eps = 0.01f;
        float phere = samplePheromone(a->x + eps, a->y);
        float pheup = samplePheromone(a->x, a->y + eps);
        dx = (phere - pheromone) / eps;
        dy = (pheup - pheromone) / eps;
        float len = sqrtf(dx*dx + dy*dy);
        if (len > 0.0001f) {
            dx = dx / len * 0.01f;
            dy = dy / len * 0.01f;
        }
    }
    
    a->vx = a->vx * 0.9f + dx * 0.1f;
    a->vy = a->vy * 0.9f + dy * 0.1f;
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // World boundaries
    if (a->x < 0) { a->x = 0; a->vx = fabsf(a->vx); }
    if (a->x > WORLD_SIZE) { a->x = WORLD_SIZE; a->vx = -fabsf(a->vx); }
    if (a->y < 0) { a->y = 0; a->vy = fabsf(a->vy); }
    if (a->y > WORLD_SIZE) { a->y = WORLD_SIZE; a->vy = -fabsf(a->vy); }
    
    // Resource interaction
    float detection_range = 0.03f + a->role[EXPLORER] * 0.04f;
    float grab_range = 0.02f + a->role[COLLECTOR] * 0.02f;
    
    Resource* nearest = nullptr;
    float nearest_dist = 999.0f;
    
    for (int i = 0; i < RESOURCE_COUNT; i++) {
        Resource* r = &resources[i];
        if (r->collected) continue;
        
        float dx = r->x - a->x;
        float dy = r->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < detection_range && dist < nearest_dist) {
            nearest = r;
            nearest_dist = dist;
        }
        
        if (dist < grab_range) {
            // Collect resource
            float bonus = 1.0f + a->role[COLLECTOR] * 0.5f;
            
            // Defender territory bonus
            int defender_count = 0;
            for (int j = 0; j < 8; j++) {
                int nidx = (idx + j * 128) % AGENT_COUNT;
                Agent* n = &agents[nidx];
                float ndist = sqrtf(powf(n->x - a->x, 2) + powf(n->y - a->y, 2));
                if (ndist < 0.1f && n->arch == DEFENDER && a->arch == DEFENDER) {
                    defender_count++;
                }
            }
            bonus += defender_count * 0.2f;
            
            a->energy += r->value * bonus;
            a->fitness += r->value * bonus;
            r->collected = 1;
            
            // Leave pheromone at resource location (novel)
            addPheromone(r->x, r->y, PHEROMONE_STRENGTH);
            
            // Communicator broadcast
            if (a->role[COMMUNICATOR] > 0.3f) {
                for (int j = 0; j < 4; j++) {
                    int nidx = (idx + j * 256) % AGENT_COUNT;
                    Agent* n = &agents[nidx];
                    float ndist = sqrtf(powf(n->x - a->x, 2) + powf(n->y - a->y, 2));
                    if (ndist < 0.06f) {
                        // Influence neighbor's movement toward resource
                        n->vx += (r->x - n->x) * 0.005f * a->role[COMMUNICATOR];
                        n->vy += (r->y - n->y) * 0.005f * a->role[COMMUNICATOR];
                    }
                }
            }
            break;
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0) {
        float resistance = a->role[DEFENDER] * 2.0f;
        if (lcgf(&a->rng) > resistance) {
            a->energy *= 0.5f;
            a->vx += lcgf(&a->rng) * 0.1f - 0.05f;
            a->vy += lcgf(&a->rng) * 0.1f - 0.05f;
        }
    }
    
    // Energy limits
    if (a->energy > 2.0f) a->energy = 2.0f;
    if (a->energy < 0) a->energy = 0;
}

int main() {
    printf("Experiment v87: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone trails as environmental memory\n");
    printf("Prediction: 20-30%% specialist efficiency improvement over v8 baseline\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENT_COUNT, RESOURCE_COUNT, TICKS);
    
    // Allocate memory
    Agent* d_agents_spec;
    Agent* d_agents_uniform;
    Resource* d_resources_spec;
    Resource* d_resources_uniform;
    
    cudaMalloc(&d_agents_spec, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_agents_uniform, AGENT_COUNT * sizeof(Agent));
    cudaMalloc(&d_resources_spec, RESOURCE_COUNT * sizeof(Resource));
    cudaMalloc(&d_resources_uniform, RESOURCE_COUNT * sizeof(Resource));
    
    Agent* h_agents_spec = new Agent[AGENT_COUNT];
    Agent* h_agents_uniform = new Agent[AGENT_COUNT];
    
    // Initialize pheromone grid
    dim3 blockSize(16, 16);
    dim3 gridSize((PHEROMONE_GRID_SIZE + 15)/16, (PHEROMONE_GRID_SIZE + 15)/16);
    initPheromone<<<gridSize, blockSize>>>();
    
    // Initialize specialized population
    init<<<(AGENT_COUNT+255)/256, 256>>>(d_agents_spec, d_resources_spec, 1);
    
    // Initialize uniform population
    init<<<(AGENT_COUNT+255)/256, 256>>>(d_agents_uniform, d_resources_uniform, 0);
    
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        // Update pheromones (diffusion and decay)
        updatePheromone<<<gridSize, blockSize>>>();
        
        // Run specialized population
        tick<<<(AGENT_COUNT+255)/256, 256>>>(d_agents_spec, d_resources_spec, t);
        
        // Run uniform population
        tick<<<(AGENT_COUNT+255)/256, 256>>>(d_agents_uniform, d_resources_uniform, t);
        
        // Respawn resources occasionally
        if (t % 50 == 0) {
            init<<<1, RESOURCE_COUNT>>>(d_agents_spec, d_resources_spec, 1);
            init<<<1, RESOURCE_COUNT>>>(d_agents_uniform, d_resources_uniform, 0);
        }
        
        if (t % 100 == 0) {
            printf("Tick %d/500\r", t);
            fflush(stdout);
        }
    }
    
    // Copy results back
    cudaMemcpy(h_agents_spec, d_agents_spec, AGENT_COUNT * sizeof(Agent), cudaMemcpyDeviceTo