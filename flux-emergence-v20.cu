/*
CUDA Simulation Experiment v20: Stigmergy with Pheromone Trails
Testing: Whether pheromone trails left at resource locations improve specialist efficiency
Prediction: Pheromones will help uniform agents more than specialists (reducing specialist advantage)
  because specialists already have optimized behaviors, while uniform agents benefit more from
  collective intelligence through stigmergy.
Baseline: v8 mechanisms (scarcity, territory, communication) + anti-convergence
Novel: Agents leave pheromone markers at collected resource sites that decay over time
  - Pheromone intensity: 1.0 at drop, decays 0.95 per tick
  - Detection: Agents sense strongest pheromone within 0.08 range
  - Influence: Agents move toward pheromone with weight 0.3 (vs random 0.7)
  - Specialists predicted to benefit less (already optimized)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define AGENTS 1024
#define RESOURCES 128
#define TICKS 500
#define ARCHES 4
#define PHEROMONES 256  // Max active pheromones

// Linear Congruential Generator
__device__ __host__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __host__ float lcgf(unsigned int &state) {
    return lcg(state) / 4294967296.0f;
}

struct Agent {
    float x, y;
    float vx, vy;
    float energy;
    float role[4];      // Role weights for each archetype
    float fitness;
    int arch;           // Dominant archetype (0-3)
    unsigned int rng;
};

struct Resource {
    float x, y;
    float value;
    int collected;
    unsigned int rng;
};

struct Pheromone {
    float x, y;
    float intensity;
    int arch;           // Which archetype left it
    int active;
};

__global__ void init(Agent *agents, Resource *resources, Pheromone *pheromones, 
                     unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < AGENTS) {
        Agent *a = &agents[idx];
        a->rng = seed + idx * 17;
        a->x = lcgf(a->rng) * 2.0f - 1.0f;
        a->y = lcgf(a->rng) * 2.0f - 1.0f;
        a->vx = lcgf(a->rng) * 0.02f - 0.01f;
        a->vy = lcgf(a->rng) * 0.02f - 0.01f;
        a->energy = 1.0f;
        a->fitness = 0.0f;
        a->arch = idx % ARCHES;
        
        // Specialized agents (first half) vs uniform control (second half)
        if (idx < AGENTS/2) {
            // Specialists: strong in their archetype role
            for (int i = 0; i < ARCHES; i++) {
                a->role[i] = (i == a->arch) ? 0.7f : 0.1f;
            }
        } else {
            // Uniform: all roles equal
            for (int i = 0; i < ARCHES; i++) {
                a->role[i] = 0.25f;
            }
        }
    }
    
    if (idx < RESOURCES) {
        Resource *r = &resources[idx];
        r->rng = seed + idx * 19;
        r->x = lcgf(r->rng) * 2.0f - 1.0f;
        r->y = lcgf(r->rng) * 2.0f - 1.0f;
        r->value = 0.5f + lcgf(r->rng) * 0.5f;
        r->collected = 0;
    }
    
    if (idx < PHEROMONES) {
        Pheromone *p = &pheromones[idx];
        p->active = 0;
        p->intensity = 0.0f;
    }
}

__device__ float distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx*dx + dy*dy);
}

__global__ void tick(Agent *agents, Resource *resources, Pheromone *pheromones, 
                     int tick_num, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= AGENTS) return;
    
    Agent *a = &agents[idx];
    
    // Energy decay
    a->energy *= 0.999f;
    
    // Anti-convergence: prevent role homogenization
    float similarity = 0.0f;
    for (int i = 0; i < ARCHES; i++) {
        similarity += fabsf(a->role[i] - 0.25f);
    }
    similarity /= ARCHES;
    
    if (similarity < 0.1f) {  // Too uniform
        int drift_idx = lcg(a->rng) % ARCHES;
        a->role[drift_idx] += (lcgf(a->rng) * 0.02f - 0.01f);
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < ARCHES; i++) sum += a->role[i];
        for (int i = 0; i < ARCHES; i++) a->role[i] /= sum;
    }
    
    // Find nearest pheromone
    float best_ph_dist = 1.0f;
    float best_ph_x = 0.0f, best_ph_y = 0.0f;
    float best_ph_intensity = 0.0f;
    
    for (int i = 0; i < PHEROMONES; i++) {
        Pheromone *p = &pheromones[i];
        if (!p->active) continue;
        
        float d = distance(a->x, a->y, p->x, p->y);
        if (d < 0.08f && p->intensity > best_ph_intensity) {
            best_ph_dist = d;
            best_ph_intensity = p->intensity;
            best_ph_x = p->x;
            best_ph_y = p->y;
        }
    }
    
    // Movement with pheromone influence (30% toward pheromone, 70% random)
    if (best_ph_intensity > 0.0f) {
        a->vx = a->vx * 0.7f + (best_ph_x - a->x) * 0.3f;
        a->vy = a->vy * 0.7f + (best_ph_y - a->y) * 0.3f;
    } else {
        // Random walk
        a->vx += lcgf(a->rng) * 0.004f - 0.002f;
        a->vy += lcgf(a->rng) * 0.004f - 0.002f;
    }
    
    // Velocity limits
    float speed = sqrtf(a->vx*a->vx + a->vy*a->vy);
    if (speed > 0.03f) {
        a->vx *= 0.03f / speed;
        a->vy *= 0.03f / speed;
    }
    
    // Update position
    a->x += a->vx;
    a->y += a->vy;
    
    // Boundary check
    if (a->x < -1.0f) { a->x = -1.0f; a->vx *= -0.5f; }
    if (a->x > 1.0f) { a->x = 1.0f; a->vx *= -0.5f; }
    if (a->y < -1.0f) { a->y = -1.0f; a->vy *= -0.5f; }
    if (a->y > 1.0f) { a->y = 1.0f; a->vy *= -0.5f; }
    
    // Role-based behaviors
    float detect_range = 0.03f + a->role[0] * 0.04f;      // Explorer role
    float grab_range = 0.02f + a->role[1] * 0.02f;        // Collector role
    float comm_range = 0.04f + a->role[2] * 0.02f;        // Communicator role
    float defend_bonus = 1.0f + a->role[3] * 0.2f;        // Defender role
    
    // Find nearest resource
    float best_dist = 1.0f;
    int best_idx = -1;
    
    for (int i = 0; i < RESOURCES; i++) {
        Resource *r = &resources[i];
        if (r->collected) continue;
        
        float d = distance(a->x, a->y, r->x, r->y);
        if (d < best_dist) {
            best_dist = d;
            best_idx = i;
        }
    }
    
    // Resource collection
    if (best_idx != -1 && best_dist < detect_range) {
        Resource *r = &resources[best_idx];
        
        if (best_dist < grab_range) {
            // Collect resource
            float value = r->value;
            if (a->role[1] > 0.3f) value *= 1.5f;  // Collector bonus
            
            // Territory bonus from nearby defenders of same archetype
            int nearby_defenders = 0;
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent *other = &agents[j];
                if (other->arch != a->arch) continue;
                if (distance(a->x, a->y, other->x, other->y) < 0.1f) {
                    nearby_defenders++;
                }
            }
            value *= (1.0f + nearby_defenders * 0.2f * a->role[3]);
            
            a->energy += value;
            a->fitness += value;
            r->collected = 1;
            
            // Leave pheromone at collection site
            for (int i = 0; i < PHEROMONES; i++) {
                Pheromone *p = &pheromones[i];
                if (!p->active) {
                    p->x = r->x;
                    p->y = r->y;
                    p->intensity = 1.0f;
                    p->arch = a->arch;
                    p->active = 1;
                    break;
                }
            }
        }
        
        // Communication behavior
        if (a->role[2] > 0.3f && best_dist < comm_range) {
            // Broadcast to nearby agents
            for (int j = 0; j < AGENTS; j++) {
                if (j == idx) continue;
                Agent *other = &agents[j];
                if (distance(a->x, a->y, other->x, other->y) < 0.06f) {
                    // Influence neighbor's velocity toward resource
                    float influence = a->role[2] * 0.1f;
                    other->vx += (r->x - other->x) * influence;
                    other->vy += (r->y - other->y) * influence;
                }
            }
        }
    }
    
    // Perturbation (every 50 ticks)
    if (tick_num % 50 == 0) {
        if (lcgf(a->rng) < 0.1f) {
            // Defenders resist perturbation
            if (a->role[3] < 0.3f) {
                a->energy *= 0.5f;
            } else {
                a->energy *= 0.8f;
            }
        }
    }
    
    // Energy limits
    if (a->energy > 2.0f) a->energy = 2.0f;
    if (a->energy < 0.0f) a->energy = 0.0f;
}

__global__ void update_pheromones(Pheromone *pheromones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PHEROMONES) return;
    
    Pheromone *p = &pheromones[idx];
    if (!p->active) return;
    
    // Decay pheromone intensity
    p->intensity *= 0.95f;
    if (p->intensity < 0.01f) {
        p->active = 0;
    }
}

__global__ void respawn_resources(Resource *resources, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RESOURCES) return;
    
    Resource *r = &resources[idx];
    if (r->collected) {
        // 10% chance to respawn each tick
        if ((lcg(r->rng) % 100) < 10) {
            r->x = lcgf(r->rng) * 2.0f - 1.0f;
            r->y = lcgf(r->rng) * 2.0f - 1.0f;
            r->value = 0.5f + lcgf(r->rng) * 0.5f;
            r->collected = 0;
        }
    }
}

int main() {
    // Allocate memory
    Agent *agents;
    Resource *resources;
    Pheromone *pheromones;
    
    cudaMallocManaged(&agents, AGENTS * sizeof(Agent));
    cudaMallocManaged(&resources, RESOURCES * sizeof(Resource));
    cudaMallocManaged(&pheromones, PHEROMONES * sizeof(Pheromone));
    
    // Initialize
    unsigned int seed = 12345;
    dim3 block(256);
    dim3 grid_agents((AGENTS + 255) / 256);
    dim3 grid_res((RESOURCES + 255) / 256);
    dim3 grid_ph((PHEROMONES + 255) / 256);
    
    init<<<grid_agents, block>>>(agents, resources, pheromones, seed);
    cudaDeviceSynchronize();
    
    // Run simulation
    for (int t = 0; t < TICKS; t++) {
        tick<<<grid_agents, block>>>(agents, resources, pheromones, t, seed + t);
        cudaDeviceSynchronize();
        
        update_pheromones<<<grid_ph, block>>>(pheromones);
        cudaDeviceSynchronize();
        
        respawn_resources<<<grid_res, block>>>(resources, seed + t + 1000);
        cudaDeviceSynchronize();
    }
    
    // Calculate results
    float spec_fitness = 0.0f, unif_fitness = 0.0f;
    float spec_energy = 0.0f, unif_energy = 0.0f;
    int spec_count = 0, unif_count = 0;
    
    for (int i = 0; i < AGENTS; i++) {
        if (i < AGENTS/2) {
            spec_fitness += agents[i].fitness;
            spec_energy += agents[i].energy;
            spec_count++;
        } else {
            unif_fitness += agents[i].fitness;
            unif_energy += agents[i].energy;
            unif_count++;
        }
    }
    
    spec_fitness /= spec_count;
    unif_fitness /= unif_count;
    spec_energy /= spec_count;
    unif_energy /= unif_count;
    
    float fitness_ratio = spec_fitness / unif_fitness;
    float energy_ratio = spec_energy / unif_energy;
    
    // Calculate specialization metric
    float avg_specialization = 0.0f;
    for (int i = 0; i < AGENTS/2; i++) {
        float max_role = 0.0f;
        for (int j = 0; j < ARCHES; j++) {
            if (agents[i].role[j] > max_role) max_role = agents[i].role[j];
        }
        avg_specialization += max_role;
    }
    avg_specialization /= (AGENTS/2);
    
    // Print results
    printf("=== EXPERIMENT v20: Stigmergy with Pheromone Trails ===\n");
    printf("Agents: %d (512 specialists + 512 uniform)\n", AGENTS);
    printf("Resources: %d (scarce)\n", RESOURCES);
    printf("Ticks: %d\n", TICKS);
    printf("Pheromones: %d max active\n", PHEROMONES);
    printf("\n");
    printf("Specialist avg fitness: %.4f\n", spec_fitness);
    printf("Uniform avg fitness:    %.4f\n", unif_fitness);
    printf("Fitness ratio (spec/unif): %.3fx\n", fitness_ratio);
    printf("\n");
    printf("Specialist avg energy: %.4f\n", spec_energy);
    printf("Uniform avg energy:    %.4f\n", unif_energy);
    printf("Energy ratio (spec/unif): %.3fx\n", energy_ratio);
    printf("\n");
    printf("Avg specialist max role: %.3f\n", avg_specialization);
   