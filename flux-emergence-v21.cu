
/*
CUDA Simulation Experiment v21: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will enhance specialist coordination beyond v8 baseline,
            especially for explore/collect roles, leading to >1.61x advantage.
Baseline: v8 mechanisms (scarcity, territory, comms) included.
Novelty: Stigmergy - agents deposit pheromones at collected resources,
         other agents sense pheromone gradients to find resources.
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define AGENTS 1024
#define RESOURCES 128
#define TICKS 500
#define ARCHETYPES 4
#define PHEROMONE_GRID_SIZE 256
#define PHEROMONE_DECAY 0.97f

// Linear congruential generator
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return (lcg(state) & 0xFFFFFF) / 16777216.0f;
}

struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];  // explore, collect, communicate, defend
    float fitness;
    int arch;
    unsigned int rng;
};

struct Resource {
    float x, y;
    float value;
    int collected;
};

struct Pheromone {
    float strength[ARCHETYPES];
};

__device__ Pheromone pheromone_grid[PHEROMONE_GRID_SIZE][PHEROMONE_GRID_SIZE];

__global__ void init_agents(Agent* agents, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    unsigned int rng = seed + idx * 17;
    
    agents[idx].x = lcgf(rng);
    agents[idx].y = lcgf(rng);
    agents[idx].vx = lcgf(rng) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(rng) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].arch = idx % ARCHETYPES;
    agents[idx].rng = rng;
    
    // Specialists: dominant role = 0.7, others 0.1
    // Uniform control: all roles = 0.25
    if (idx < AGENTS/2) {  // First half: specialists
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.1f;
        }
        agents[idx].role[agents[idx].arch] = 0.7f;
    } else {  // Second half: uniform control
        for (int i = 0; i < 4; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

__global__ void init_resources(Resource* resources, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    unsigned int rng = seed + idx * 19;
    resources[idx].x = lcgf(rng);
    resources[idx].y = lcgf(rng);
    resources[idx].value = 0.5f + lcgf(rng) * 0.5f;
    resources[idx].collected = 0;
}

__global__ void init_pheromones() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < PHEROMONE_GRID_SIZE && y < PHEROMONE_GRID_SIZE) {
        for (int a = 0; a < ARCHETYPES; a++) {
            pheromone_grid[x][y].strength[a] = 0.0f;
        }
    }
}

__device__ float similarity(const Agent& a, const Agent& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < 4; i++) {
        dot += a.role[i] * b.role[i];
        norm_a += a.role[i] * a.role[i];
        norm_b += b.role[i] * b.role[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}

__device__ void apply_anti_convergence(Agent& agent, unsigned int& rng) {
    float max_role = 0.0f;
    int dominant = 0;
    for (int i = 0; i < 4; i++) {
        if (agent.role[i] > max_role) {
            max_role = agent.role[i];
            dominant = i;
        }
    }
    
    // Apply random drift to non-dominant roles
    for (int i = 0; i < 4; i++) {
        if (i != dominant) {
            agent.role[i] += (lcgf(rng) - 0.5f) * 0.01f;
            agent.role[i] = fmaxf(0.05f, fminf(0.8f, agent.role[i]));
        }
    }
    
    // Renormalize
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) sum += agent.role[i];
    for (int i = 0; i < 4; i++) agent.role[i] /= sum;
}

__device__ void deposit_pheromone(float x, float y, int arch, float amount) {
    int grid_x = (int)(x * PHEROMONE_GRID_SIZE);
    int grid_y = (int)(y * PHEROMONE_GRID_SIZE);
    
    if (grid_x >= 0 && grid_x < PHEROMONE_GRID_SIZE &&
        grid_y >= 0 && grid_y < PHEROMONE_GRID_SIZE) {
        atomicAdd(&pheromone_grid[grid_x][grid_y].strength[arch], amount);
    }
}

__device__ float sample_pheromone(float x, float y, int arch) {
    int grid_x = (int)(x * PHEROMONE_GRID_SIZE);
    int grid_y = (int)(y * PHEROMONE_GRID_SIZE);
    
    if (grid_x >= 0 && grid_x < PHEROMONE_GRID_SIZE &&
        grid_y >= 0 && grid_y < PHEROMONE_GRID_SIZE) {
        return pheromone_grid[grid_x][grid_y].strength[arch];
    }
    return 0.0f;
}

__global__ void tick(Agent* agents, Resource* resources, int tick_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent& a = agents[idx];
    
    // Energy decay
    a.energy *= 0.999f;
    if (a.energy <= 0.001f) {
        a.energy = 0.001f;
        return;
    }
    
    // Perturbation (10% chance every 50 ticks)
    if (tick_num % 50 == 0 && lcgf(a.rng) < 0.1f) {
        if (a.role[3] < 0.3f) {  // Weak defenders get full penalty
            a.energy *= 0.5f;
        } else {  // Strong defenders resist
            a.energy *= 0.8f;
        }
    }
    
    // Sense pheromone gradient
    float current_pheromone = sample_pheromone(a.x, a.y, a.arch);
    float dx = 0.0f, dy = 0.0f;
    
    // Sample 4 directions for gradient
    float offsets[4][2] = {{0.01f,0.0f}, {-0.01f,0.0f}, {0.0f,0.01f}, {0.0f,-0.01f}};
    for (int i = 0; i < 4; i++) {
        float px = a.x + offsets[i][0];
        float py = a.y + offsets[i][1];
        if (px < 0.0f || px > 1.0f || py < 0.0f || py > 1.0f) continue;
        
        float p = sample_pheromone(px, py, a.arch);
        if (p > current_pheromone) {
            dx += offsets[i][0] * (p - current_pheromone);
            dy += offsets[i][1] * (p - current_pheromone);
        }
    }
    
    // Normalize pheromone influence
    float norm = sqrtf(dx*dx + dy*dy) + 1e-8f;
    dx /= norm;
    dy /= norm;
    
    // Movement: blend random walk with pheromone following
    float explore_strength = a.role[0] * 2.0f;  // Explore role amplifies pheromone following
    a.vx = a.vx * 0.8f + (lcgf(a.rng) - 0.5f) * 0.02f + dx * explore_strength * 0.03f;
    a.vy = a.vy * 0.8f + (lcgf(a.rng) - 0.5f) * 0.02f + dy * explore_strength * 0.03f;
    
    // Limit velocity
    float speed = sqrtf(a.vx*a.vx + a.vy*a.vy);
    if (speed > 0.03f) {
        a.vx *= 0.03f / speed;
        a.vy *= 0.03f / speed;
    }
    
    // Update position
    a.x += a.vx;
    a.y += a.vy;
    
    // Boundary wrap
    if (a.x < 0.0f) a.x = 1.0f + a.x;
    if (a.x > 1.0f) a.x = a.x - 1.0f;
    if (a.y < 0.0f) a.y = 1.0f + a.y;
    if (a.y > 1.0f) a.y = a.y - 1.0f;
    
    // Resource interaction
    float best_dist = 1.0f;
    int best_res = -1;
    
    for (int r = 0; r < RESOURCES; r++) {
        Resource& res = resources[r];
        if (res.collected) continue;
        
        float dx = a.x - res.x;
        float dy = a.y - res.y;
        // Wrap-around distance
        dx = fminf(fabsf(dx), fminf(fabsf(dx-1.0f), fabsf(dx+1.0f)));
        dy = fminf(fabsf(dy), fminf(fabsf(dy-1.0f), fabsf(dy+1.0f)));
        float dist = sqrtf(dx*dx + dy*dy);
        
        // Detection range based on explore role
        if (dist < 0.03f + a.role[0] * 0.04f && dist < best_dist) {
            best_dist = dist;
            best_res = r;
        }
        
        // Collection range based on collect role
        if (dist < 0.02f + a.role[1] * 0.02f) {
            // Collect resource
            float bonus = 1.0f + a.role[1] * 0.5f;  // Collectors get bonus
            
            // Territory bonus from nearby defenders of same archetype
            int nearby_defenders = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent& other = agents[j];
                if (other.arch != a.arch) continue;
                if (other.role[3] < 0.3f) continue;
                
                float odx = a.x - other.x;
                float ody = a.y - other.y;
                odx = fminf(fabsf(odx), fminf(fabsf(odx-1.0f), fabsf(odx+1.0f)));
                ody = fminf(fabsf(ody), fminf(fabsf(ody-1.0f), fabsf(ody+1.0f)));
                if (sqrtf(odx*odx + ody*ody) < 0.1f) {
                    nearby_defenders++;
                }
            }
            float territory_bonus = 1.0f + nearby_defenders * 0.2f;
            
            float gain = res.value * bonus * territory_bonus;
            a.energy += gain;
            a.fitness += gain;
            res.collected = 1;
            
            // Deposit pheromone at resource location
            deposit_pheromone(res.x, res.y, a.arch, 1.0f + a.role[0] * 2.0f);
            
            break;
        }
    }
    
    // Communication
    if (a.role[2] > 0.3f && best_res != -1) {
        Resource& res = resources[best_res];
        for (int j = 0; j < AGENTS; j++) {
            if (j == idx) continue;
            Agent& other = agents[j];
            if (other.arch != a.arch) continue;
            
            float dx = a.x - other.x;
            float dy = a.y - other.y;
            dx = fminf(fabsf(dx), fminf(fabsf(dx-1.0f), fabsf(dx+1.0f)));
            dy = fminf(fabsf(dy), fminf(fabsf(dy-1.0f), fabsf(dy+1.0f)));
            
            if (sqrtf(dx*dx + dy*dy) < 0.06f) {
                // Communicate resource location
                float influence = a.role[2] * 0.5f;
                other.vx += (res.x - other.x) * influence * 0.01f;
                other.vy += (res.y - other.y) * influence * 0.01f;
            }
        }
    }
    
    // Social learning with anti-convergence
    if (lcgf(a.rng) < 0.1f) {
        int partner = lcgf(a.rng) * AGENTS;
        if (partner != idx) {
            Agent& other = agents[partner];
            float sim = similarity(a, other);
            float coupling = (a.arch == other.arch) ? 0.02f : 0.002f;
            
            if (sim > 0.9f) {
                apply_anti_convergence(a, a.rng);
            } else {
                for (int i = 0; i < 4; i++) {
                    a.role[i] += (other.role[i] - a.role[i]) * coupling;
                }
            }
        }
    }
}

__global__ void decay_pheromones() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < PHEROMONE_GRID_SIZE && y < PHEROMONE_GRID_SIZE) {
        for (int a = 0; a < ARCHETYPES; a++) {
            pheromone_grid[x][y].strength[a] *= PHEROMONE_DECAY;
        }
    }
}

__global__ void respawn_resources(Resource* resources, int tick_num, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    if (tick_num % 50 == 0) {
        resources[idx].collected = 0;
        unsigned int rng = seed + idx * 23 + tick_num;
        if (lcgf(rng) < 0.3f) {
            resources[idx].x = lcgf(rng);
            resources[idx].y = lcgf(rng);
        }
    }
}

int main() {
    printf("Experiment v21: Stigmergy with Pheromone Trails\n");
    printf("Testing: Pheromone markers enhance specialist coordination\n");
    printf("Prediction: Specialist advantage >1.61x (v8 baseline)\n");
    printf("Agents: %d (512 specialists + 512 uniform)\n", AGENTS);
    printf("Resources: %d, Ticks: %d\n\n", RESOURCES, TICKS);
    
    // Allocate memory
    Agent* agents;
    Resource* resources;
    cudaMallocManaged(&agents, AGENTS * sizeof(Agent));
    cudaMallocManaged(&resources, RESOURCES * sizeof(Resource));
    
    // Initialize
    init_agents<<<(AGENTS+255)/256, 256>>>(agents, 12345);
    init_resources<<<(RESOURCES+255)/256, 256>>>(resources, 67890);
    
    dim3 grid_size((PHEROMONE_GRID_SIZE+15)/16, (PHEROMONE_GRID_SIZE+15)/16);
    dim3 block_size(16, 16);
   