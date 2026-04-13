
/*
CUDA Simulation Experiment v86: Stigmergy with Pheromone Trails
Testing: Agents leave pheromone markers at resource locations that decay over time.
Prediction: Pheromones will improve collective efficiency for specialists more than 
uniform agents, increasing the specialist advantage ratio beyond 1.61x.
Baseline: v8 mechanisms (scarcity, territory, comms) + anti-convergence.
Novel: Pheromone trails that agents can detect and follow.
*/
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Constants
const int AGENTS = 1024;
const int RESOURCES = 128;
const int TICKS = 500;
const int ARCHETYPES = 4;
const float WORLD_SIZE = 1.0f;
const float ENERGY_DECAY = 0.999f;
const float PERTURB_ENERGY_FACTOR = 0.5f;
const float PERTURB_PROB = 0.001f;
const float DRIFT_STRENGTH = 0.01f;
const float SIMILARITY_THRESHOLD = 0.9f;
const float SAME_ARCH_COUPLING = 0.02f;
const float DIFF_ARCH_COUPLING = 0.002f;
const float DEFEND_BOOST_PER_DEFENDER = 0.2f;
const float PHEROMONE_DECAY = 0.95f;
const float PHEROMONE_DETECTION_RANGE = 0.08f;
const float PHEROMONE_STRENGTH_INITIAL = 1.0f;

// RNG functions
__device__ __host__ unsigned int lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__device__ __host__ float lcgf(unsigned int* state) {
    return lcg(state) / 4294967296.0f;
}

// Resource with pheromone field
struct Resource {
    float x, y;
    float value;
    bool collected;
    float pheromone;
    unsigned int rng_state;
};

// Agent
struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[ARCHETYPES];  // explore, collect, communicate, defend
    float fitness;
    int arch;
    unsigned int rng;
};

// Pheromone marker
struct Pheromone {
    float x, y;
    float strength;
    int arch_source;
};

// Global device pointers
__device__ Agent* d_agents;
__device__ Resource* d_resources;
__device__ Pheromone* d_pheromones;
__device__ int* d_pheromone_count;
__device__ int pheromone_capacity;

// Kernel to initialize agents
__global__ void initAgents(Agent* agents, bool specialized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    agents[idx].x = lcgf(&agents[idx].rng) * WORLD_SIZE;
    agents[idx].y = lcgf(&agents[idx].rng) * WORLD_SIZE;
    agents[idx].vx = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
    agents[idx].vy = lcgf(&agents[idx].rng) * 0.02f - 0.01f;
    agents[idx].energy = 1.0f;
    agents[idx].fitness = 0.0f;
    agents[idx].arch = idx % ARCHETYPES;
    agents[idx].rng = idx * 12345 + 1;
    
    if (specialized) {
        // Specialists: high value in own archetype role
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = (i == agents[idx].arch) ? 0.7f : 0.1f;
        }
    } else {
        // Uniform: all roles equal
        for (int i = 0; i < ARCHETYPES; i++) {
            agents[idx].role[i] = 0.25f;
        }
    }
}

// Kernel to initialize resources
__global__ void initResources(Resource* resources) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    resources[idx].rng_state = idx * 6789 + 1;
    resources[idx].x = lcgf(&resources[idx].rng_state) * WORLD_SIZE;
    resources[idx].y = lcgf(&resources[idx].rng_state) * WORLD_SIZE;
    resources[idx].value = 0.5f + lcgf(&resources[idx].rng_state) * 0.5f;
    resources[idx].collected = false;
    resources[idx].pheromone = 0.0f;
}

// Kernel to add pheromone marker
__device__ void addPheromone(float x, float y, float strength, int arch) {
    int idx = atomicAdd(d_pheromone_count, 1);
    if (idx < pheromone_capacity) {
        d_pheromones[idx].x = x;
        d_pheromones[idx].y = y;
        d_pheromones[idx].strength = strength;
        d_pheromones[idx].arch_source = arch;
    }
}

// Kernel to decay and remove weak pheromones
__global__ void updatePheromones() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *d_pheromone_count) return;
    
    d_pheromones[idx].strength *= PHEROMONE_DECAY;
}

// Kernel to clean up weak pheromones
__global__ void cleanupPheromones() {
    // Simple approach: reset if count gets too high
    if (*d_pheromone_count > pheromone_capacity * 0.9) {
        *d_pheromone_count = 0;
    }
}

// Main simulation tick kernel
__global__ void tick(int step, float* total_fitness_specialized, float* total_fitness_uniform, 
                     bool is_specialized_group) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent* a = &d_agents[idx];
    
    // Energy decay
    a->energy *= ENERGY_DECAY;
    
    // Anti-convergence: check similarity with neighbors
    int similar_count = 0;
    int total_count = 0;
    for (int i = 0; i < AGENTS; i++) {
        if (i == idx) continue;
        Agent* other = &d_agents[i];
        float dx = a->x - other->x;
        float dy = a->y - other->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < 0.1f) {
            total_count++;
            float role_sim = 0.0f;
            for (int r = 0; r < ARCHETYPES; r++) {
                role_sim += fabsf(a->role[r] - other->role[r]);
            }
            role_sim = 1.0f - role_sim / ARCHETYPES;
            if (role_sim > SIMILARITY_THRESHOLD) similar_count++;
        }
    }
    
    // Apply drift if too similar
    if (total_count > 0 && (float)similar_count / total_count > 0.5f) {
        int dominant_role = 0;
        for (int r = 1; r < ARCHETYPES; r++) {
            if (a->role[r] > a->role[dominant_role]) dominant_role = r;
        }
        int drift_role = (dominant_role + 1) % ARCHETYPES;
        a->role[drift_role] += (lcgf(&a->rng) * 2.0f - 1.0f) * DRIFT_STRENGTH;
        a->role[drift_role] = fmaxf(0.0f, fminf(1.0f, a->role[drift_role]));
    }
    
    // Normalize roles
    float role_sum = a->role[0] + a->role[1] + a->role[2] + a->role[3];
    for (int r = 0; r < ARCHETYPES; r++) {
        a->role[r] /= role_sum;
    }
    
    // Role-based behavior
    float explore_strength = a->role[0];
    float collect_strength = a->role[1];
    float comm_strength = a->role[2];
    float defend_strength = a->role[3];
    
    // Explore behavior: random movement with some bias
    if (explore_strength > 0.1f) {
        a->vx += (lcgf(&a->rng) * 2.0f - 1.0f) * 0.005f * explore_strength;
        a->vy += (lcgf(&a->rng) * 2.0f - 1.0f) * 0.005f * explore_strength;
    }
    
    // Pheromone following: move toward strongest pheromone of same archetype
    float best_pheromone_x = 0.0f, best_pheromone_y = 0.0f;
    float best_strength = 0.0f;
    int pheromone_count = *d_pheromone_count;
    
    for (int p = 0; p < pheromone_count; p++) {
        Pheromone* ph = &d_pheromones[p];
        if (ph->arch_source != a->arch) continue;
        
        float dx = ph->x - a->x;
        float dy = ph->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        if (dist < PHEROMONE_DETECTION_RANGE && ph->strength > best_strength) {
            best_strength = ph->strength;
            best_pheromone_x = ph->x;
            best_pheromone_y = ph->y;
        }
    }
    
    if (best_strength > 0.0f) {
        float dx = best_pheromone_x - a->x;
        float dy = best_pheromone_y - a->y;
        float dist = sqrtf(dx*dx + dy*dy) + 0.0001f;
        a->vx += (dx / dist) * 0.01f * explore_strength;
        a->vy += (dy / dist) * 0.01f * explore_strength;
    }
    
    // Resource detection and collection
    float nearest_res_x = 0.0f, nearest_res_y = 0.0f;
    float nearest_dist = 1.0f;
    int nearest_idx = -1;
    
    for (int r = 0; r < RESOURCES; r++) {
        Resource* res = &d_resources[r];
        if (res->collected) continue;
        
        float dx = res->x - a->x;
        float dy = res->y - a->y;
        float dist = sqrtf(dx*dx + dy*dy);
        
        float detect_range = 0.03f + 0.04f * explore_strength;
        if (dist < detect_range && dist < nearest_dist) {
            nearest_dist = dist;
            nearest_res_x = res->x;
            nearest_res_y = res->y;
            nearest_idx = r;
        }
    }
    
    // Move toward nearest resource
    if (nearest_idx != -1) {
        float dx = nearest_res_x - a->x;
        float dy = nearest_res_y - a->y;
        float dist = sqrtf(dx*dx + dy*dy) + 0.0001f;
        a->vx += (dx / dist) * 0.02f * collect_strength;
        a->vy += (dy / dist) * 0.02f * collect_strength;
        
        // Collect if in range
        float grab_range = 0.02f + 0.02f * collect_strength;
        if (nearest_dist < grab_range) {
            Resource* res = &d_resources[nearest_idx];
            float value = res->value * (1.0f + 0.5f * collect_strength);
            a->energy += value;
            a->fitness += value;
            res->collected = true;
            
            // Leave pheromone at resource location
            addPheromone(res->x, res->y, PHEROMONE_STRENGTH_INITIAL, a->arch);
        }
    }
    
    // Communication behavior
    if (comm_strength > 0.1f && nearest_idx != -1) {
        float broadcast_range = 0.06f;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &d_agents[i];
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < broadcast_range) {
                // Influence neighbor's velocity toward resource
                float influence = comm_strength * 0.01f;
                float rdx = nearest_res_x - other->x;
                float rdy = nearest_res_y - other->y;
                float rdist = sqrtf(rdx*rdx + rdy*rdy) + 0.0001f;
                other->vx += (rdx / rdist) * influence;
                other->vy += (rdy / rdist) * influence;
            }
        }
    }
    
    // Defense behavior: territory and perturbation resistance
    if (defend_strength > 0.1f) {
        // Count nearby defenders of same archetype
        int defender_count = 0;
        for (int i = 0; i < AGENTS; i++) {
            if (i == idx) continue;
            Agent* other = &d_agents[i];
            if (other->arch != a->arch) continue;
            
            float dx = other->x - a->x;
            float dy = other->y - a->y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < 0.08f && other->role[3] > 0.3f) {
                defender_count++;
            }
        }
        
        // Defense boost
        float boost = 1.0f + defender_count * DEFEND_BOOST_PER_DEFENDER * defend_strength;
        a->energy *= boost;
        a->fitness *= boost;
    }
    
    // Perturbation (simulated environmental stress)
    if (lcgf(&a->rng) < PERTURB_PROB) {
        float resistance = 1.0f - defend_strength * 0.5f;
        a->energy *= PERTURB_ENERGY_FACTOR * resistance;
        a->vx += (lcgf(&a->rng) * 2.0f - 1.0f) * 0.1f;
        a->vy += (lcgf(&a->rng) * 2.0f - 1.0f) * 0.1f;
    }
    
    // Update position with velocity damping
    a->vx *= 0.95f;
    a->vy *= 0.95f;
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary wrap
    if (a->x < 0) a->x += WORLD_SIZE;
    if (a->x > WORLD_SIZE) a->x -= WORLD_SIZE;
    if (a->y < 0) a->y += WORLD_SIZE;
    if (a->y > WORLD_SIZE) a->y -= WORLD_SIZE;
    
    // Update fitness total
    if (is_specialized_group) {
        atomicAdd(total_fitness_specialized, a->fitness);
    } else {
        atomicAdd(total_fitness_uniform, a->fitness);
    }
}

// Kernel to respawn resources
__global__ void respawnResources() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource* res = &d_resources[idx];
    if (res->collected) {
        res->x = lcgf(&res->rng_state) * WORLD_SIZE;
        res->y = lcgf(&res->rng_state) * WORLD_SIZE;
        res->value = 0.5f + lcgf(&res->rng_state) * 0.5f;
        res->collected = false;
        res->pheromone = 0.0f;
    }
}

int main() {
    printf("Experiment v86: Stigmergy with Pheromone Trails\n");
    printf("Testing if pheromone trails increase specialist advantage beyond baseline 1.61x\n");
    printf("Agents: %d, Resources: %d, Ticks: %d\n\n", AGENTS, RESOURCES, TICKS);
    
    // Allocate host memory
    Agent* h_agents_specialized = new Agent[AGENTS];
    Agent* h_agents_uniform = new Agent[AGENTS];
    Resource* h_resources = new Resource[RESOURCES];
    
    // Allocate device memory
    Agent* d_agents_specialized;
    Agent* d_agents_uniform;
    Resource* d_resources;
    float* d_fitness_specialized;
    float* d_fitness_uniform;
    
    cudaMalloc(&d_agents_specialized, sizeof(