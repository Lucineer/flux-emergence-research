
/*
CUDA Simulation Experiment v52: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at collected resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination, increasing their advantage over uniform agents.
Baseline: v8 mechanisms (scarcity, territory, communication) included.
Comparison: Specialized (role[arch]=0.7) vs Uniform (all roles=0.25).
Expected: Specialists will use pheromones more effectively due to role differentiation.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int PHEROMONE_GRID_SIZE = 64;
const float WORLD_SIZE = 1.0f;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_DIFFUSE = 0.1f;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

// Pheromone struct
struct Pheromone {
    float value;
    int type; // 0: explore, 1: collect, 2: communicate, 3: defend
};

// Agent struct
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];
    float fitness;
    int arch;
    unsigned int rng;
};

// Resource struct
struct Resource {
    float x, y;
    float value;
    int collected;
};

// Global device arrays
__device__ Agent d_agents[AGENTS];
__device__ Resource d_resources[RESOURCES];
__device__ Pheromone d_pheromones[PHEROMONE_GRID_SIZE][PHEROMONE_GRID_SIZE];
__device__ int d_collected_specialized = 0;
__device__ int d_collected_uniform = 0;

// Initialize pheromone grid
__global__ void initPheromones() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < PHEROMONE_GRID_SIZE * PHEROMONE_GRID_SIZE; i += stride) {
        int x = i % PHEROMONE_GRID_SIZE;
        int y = i / PHEROMONE_GRID_SIZE;
        d_pheromones[y][x].value = 0.0f;
        d_pheromones[y][x].type = -1;
    }
}

// Initialize agents
__global__ void initAgents() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent &a = d_agents[idx];
    a.rng = 12345 + idx * 6789;
    a.x = lcgf(a.rng);
    a.y = lcgf(a.rng);
    a.vx = lcgf(a.rng) * 0.02f - 0.01f;
    a.vy = lcgf(a.rng) * 0.02f - 0.01f;
    a.energy = 1.0f;
    a.fitness = 0.0f;
    a.arch = idx % 4; // 4 archetypes
    
    // Specialized agents (first half): role[arch]=0.7, others 0.1
    // Uniform agents (second half): all roles=0.25
    if (idx < AGENTS/2) {
        for (int i = 0; i < 4; i++) a.role[i] = 0.1f;
        a.role[a.arch] = 0.7f;
    } else {
        for (int i = 0; i < 4; i++) a.role[i] = 0.25f;
    }
}

// Initialize resources
__global__ void initResources() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = d_resources[idx];
    unsigned int rng = 4567 + idx * 8910;
    r.x = lcgf(rng);
    r.y = lcgf(rng);
    r.value = 0.8f + lcgf(rng) * 0.4f;
    r.collected = 0;
}

// Update pheromone diffusion and decay
__global__ void updatePheromones() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= PHEROMONE_GRID_SIZE || y >= PHEROMONE_GRID_SIZE) return;
    
    // Shared memory for diffusion
    __shared__ float shared[18][18]; // 16x16 block with halo
    
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load center
    if (x < PHEROMONE_GRID_SIZE && y < PHEROMONE_GRID_SIZE) {
        shared[ty][tx] = d_pheromones[y][x].value;
    }
    
    // Load halo edges
    if (threadIdx.x == 0) {
        int left = max(x - 1, 0);
        shared[ty][0] = d_pheromones[y][left].value;
    }
    if (threadIdx.x == 15) {
        int right = min(x + 1, PHEROMONE_GRID_SIZE - 1);
        shared[ty][17] = d_pheromones[y][right].value;
    }
    if (threadIdx.y == 0) {
        int top = max(y - 1, 0);
        shared[0][tx] = d_pheromones[top][x].value;
    }
    if (threadIdx.y == 15) {
        int bottom = min(y + 1, PHEROMONE_GRID_SIZE - 1);
        shared[17][tx] = d_pheromones[bottom][x].value;
    }
    
    __syncthreads();
    
    // Diffuse
    float sum = shared[ty][tx] * (1.0f - 4.0f * PHEROMONE_DIFFUSE);
    sum += PHEROMONE_DIFFUSE * (shared[ty][tx-1] + shared[ty][tx+1] + 
                                shared[ty-1][tx] + shared[ty+1][tx]);
    
    // Decay
    sum *= PHEROMONE_DECAY;
    
    if (x < PHEROMONE_GRID_SIZE && y < PHEROMONE_GRID_SIZE) {
        d_pheromones[y][x].value = sum;
        if (sum < 0.001f) d_pheromones[y][x].type = -1;
    }
}

// Deposit pheromone at location
__device__ void depositPheromone(float x, float y, int type) {
    int gx = (int)(x * PHEROMONE_GRID_SIZE);
    int gy = (int)(y * PHEROMONE_GRID_SIZE);
    gx = max(0, min(PHEROMONE_GRID_SIZE - 1, gx));
    gy = max(0, min(PHEROMONE_GRID_SIZE - 1, gy));
    
    atomicAdd(&d_pheromones[gy][gx].value, 0.5f);
    d_pheromones[gy][gx].type = type;
}

// Read pheromone gradient
__device__ void sensePheromone(float x, float y, int preferred_type, float &dx, float &dy) {
    int gx = (int)(x * PHEROMONE_GRID_SIZE);
    int gy = (int)(y * PHEROMONE_GRID_SIZE);
    gx = max(0, min(PHEROMONE_GRID_SIZE - 1, gx));
    gy = max(0, min(PHEROMONE_GRID_SIZE - 1, gy));
    
    dx = dy = 0.0f;
    float center = 0.0f;
    
    // Sample 3x3 neighborhood
    for (int ox = -1; ox <= 1; ox++) {
        for (int oy = -1; oy <= 1; oy++) {
            int sx = gx + ox;
            int sy = gy + oy;
            if (sx < 0 || sx >= PHEROMONE_GRID_SIZE || sy < 0 || sy >= PHEROMONE_GRID_SIZE) continue;
            
            float weight = d_pheromones[sy][sx].value;
            if (d_pheromones[sy][sx].type == preferred_type) weight *= 2.0f;
            
            if (ox == 0 && oy == 0) {
                center = weight;
            } else {
                dx += weight * ox;
                dy += weight * oy;
            }
        }
    }
    
    // Normalize
    float len = sqrtf(dx*dx + dy*dy);
    if (len > 0.0f) {
        dx /= len;
        dy /= len;
    }
}

// Main simulation tick
__global__ void tick(int step) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENTS) return;
    
    Agent &a = d_agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    
    // Anti-convergence: detect similarity > 0.9, drift non-dominant role
    float max_role = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < 4; i++) {
        if (a.role[i] > max_role) {
            max_role = a.role[i];
            max_idx = i;
        }
    }
    
    if (max_role > 0.9f) {
        int drift_idx = (max_idx + 1 + (a.rng % 3)) % 4;
        a.role[drift_idx] += (lcgf(a.rng) * 0.02f - 0.01f);
        // Renormalize
        float sum = a.role[0] + a.role[1] + a.role[2] + a.role[3];
        for (int i = 0; i < 4; i++) a.role[i] /= sum;
    }
    
    // Pheromone sensing (specialists follow their archetype's pheromones)
    float pheromone_dx = 0.0f, pheromone_dy = 0.0f;
    if (idx < AGENTS/2) { // Specialists only
        sensePheromone(a.x, a.y, a.arch, pheromone_dx, pheromone_dy);
    }
    
    // Movement with pheromone influence
    a.vx += pheromone_dx * 0.005f * a.role[a.arch];
    a.vy += pheromone_dy * 0.005f * a.role[a.arch];
    
    // Damping
    a.vx *= 0.98f;
    a.vy *= 0.98f;
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // World boundaries
    if (a.x < 0.0f) { a.x = 0.0f; a.vx = fabsf(a.vx); }
    if (a.x > WORLD_SIZE) { a.x = WORLD_SIZE; a.vx = -fabsf(a.vx); }
    if (a.y < 0.0f) { a.y = 0.0f; a.vy = fabsf(a.vy); }
    if (a.y > WORLD_SIZE) { a.y = WORLD_SIZE; a.vy = -fabsf(a.vy); }
    
    // Resource interaction
    for (int r = 0; r < RESOURCES; r++) {
        Resource &res = d_resources[r];
        if (res.collected) continue;
        
        float dx = res.x - a.x;
        float dy = res.y - a.y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        // Detection range based on explore role
        if (dist < 0.03f + a.role[0] * 0.04f) {
            // Collect range based on collect role
            if (dist < 0.02f + a.role[1] * 0.02f) {
                // Collect resource
                float bonus = 1.0f + a.role[1] * 0.5f; // Collect bonus
                
                // Territory bonus: defenders nearby
                int defenders_nearby = 0;
                for (int j = 0; j < AGENTS; j++) {
                    if (j == idx) continue;
                    Agent &other = d_agents[j];
                    float odx = other.x - a.x;
                    float ody = other.y - a.y;
                    if (sqrtf(odx*odx + ody*ody) < 0.08f && other.arch == 3) {
                        defenders_nearby++;
                    }
                }
                bonus += defenders_nearby * 0.2f;
                
                // Collect
                a.energy += res.value * bonus;
                a.fitness += res.value * bonus;
                res.collected = 1;
                
                // Count collection for statistics
                if (idx < AGENTS/2) {
                    atomicAdd(&d_collected_specialized, 1);
                } else {
                    atomicAdd(&d_collected_uniform, 1);
                }
                
                // NOVEL MECHANISM: Deposit pheromone at collected location
                depositPheromone(res.x, res.y, a.arch);
                
                // Communication: broadcast location to nearby agents
                for (int j = 0; j < AGENTS; j++) {
                    if (j == idx) continue;
                    Agent &other = d_agents[j];
                    float odx = other.x - a.x;
                    float ody = other.y - a.y;
                    if (sqrtf(odx*odx + ody*ody) < 0.06f && a.role[2] > 0.3f) {
                        // Influence neighbor's velocity toward resource
                        other.vx += dx * 0.01f * a.role[2];
                        other.vy += dy * 0.01f * a.role[2];
                    }
                }
                break;
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (step % 50 == 25) {
        // Defenders resist perturbation
        if (a.arch != 3 || lcgf(a.rng) > 0.7f) {
            a.energy *= 0.5f;
            a.vx += lcgf(a.rng) * 0.1f - 0.05f;
            a.vy += lcgf(a.rng) * 0.1f - 0.05f;
        }
    }
    
    // Coupling: align with similar agents
    for (int j = 0; j < AGENTS; j++) {
        if (j == idx) continue;
        Agent &other = d_agents[j];
        float odx = other.x - a.x;
        float ody = other.y - a.y;
        float dist = sqrtf(odx*odx + ody*ody);
        
        if (dist < 0.08f) {
            float coupling = (a.arch == other.arch) ? 0.02f : 0.002f;
            a.vx += odx * coupling;
            a.vy += ody * coupling;
        }
    }
}

// Reset resources periodically
__global__ void resetResources() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= RESOURCES) return;
    
    Resource &r = d_resources[idx];
    if (r.collected) {
        unsigned int rng = 4567 + idx * 8910 + 12345;
        r.x = lcgf(rng);
        r.y = lcgf(rng);
        r.value = 0.8f + lcgf(rng) * 0.4f;
        r.collected = 0;
    }
}

int main() {
    printf("Experiment v52: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone markers enhance specialist coordination\n");
    printf("Prediction: Specialists >1.61x advantage over uniform agents\n");
    printf("Agents: %d (512 specialized, 512 uniform)\n", AGENTS);
    printf("Resources: %d, Ticks: %d\n", RESOURCES, TICKS);
    printf("Pheromone grid: %dx%d, Decay: %.2f, Diffuse: %.2f\n\n", 
           PHEROMONE_GRID_SIZE, PHEROMONE_GRID_SIZE, PHEROMONE_DECAY, PHEROMONE_DIFFUSE);
    
    // Initialize
    initPheromones<<<dim3(4,4), dim3(16,16)>>>();
    initAgents<<<(AGENTS+255)/256, 256>>>();
    initResources<<<(RESOURCES+255)/256,