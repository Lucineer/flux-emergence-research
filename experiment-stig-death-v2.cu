// Experiment: Stigmergy + Death Observation Synergy (Fixed)
// Laws 188-189: Stigmergy marks = high-density food zones, not eaten food
// Key fix: mark WHERE food is dense, not WHERE food was eaten

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 256
#define AGENTS 256
#define FOOD 300
#define STEPS 3000
#define BLK 128
#define NMarks 500

typedef struct { float x,y; int alive; float energy; } Agent;
typedef struct { float x,y; int active; } Food;
typedef struct { float x,y; float str; } Mark;

__device__ static float mn(float a,float b) { return a<b?a:b; }
__device__ static float mx(float a,float b) { return a>b?a:b; }
__device__ static float cr2(int *seed) { *seed = (*seed * 1103515245 + 12345) & 0x7fffffff; return (float)*seed / 0x7fffffff; }
__device__ static float wrap(float v) { return v - floorf(v/(float)N)*(float)N; }
__device__ static float dist(float x1,float y1,float x2,float y2) {
    float dx=abs(x1-x2); float dy=abs(y1-y2);
    dx=mn(dx,N-dx); dy=mn(dy,N-dy); return sqrtf(dx*dx+dy*dy);
}

__global__ void init_all(Agent *ag, Food *fd, Mark *mk, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<AGENTS) {
        int s=seed+i*137;
        ag[i].x=cr2(&s)*(float)N; ag[i].y=cr2(&s)*(float)N;
        ag[i].alive=1; ag[i].energy=50.0f;
    }
    if(i<FOOD) {
        int s=seed+i*251+999;
        fd[i].x=cr2(&s)*(float)N; fd[i].y=cr2(&s)*(float)N; fd[i].active=1;
    }
    if(i<NMarks) { mk[i].str=0; }
}

__global__ void step(Agent *ag, Food *fd, Mark *mk,
                     int use_stig, int use_death, float grab,
                     int *death_x, int *death_y, int *ndeath,
                     int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=AGENTS || !ag[i].alive) return;
    
    int s=seed+i*31;
    float r1=cr2(&s), r2=cr2(&s);
    
    // Count nearby food (density sensing)
    int nearby_food = 0;
    float cx=0, cy=0;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        float d = dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y);
        if(d < grab*5.0f) {
            nearby_food++;
            cx += fd[j].x; cy += fd[j].y;
        }
    }
    
    // Target: random walk base
    float tx = ag[i].x + (r1-0.5f)*10.0f;
    float ty = ag[i].y + (r2-0.5f)*10.0f;
    
    // Stigmergy: navigate toward marked high-density zones
    if(use_stig) {
        float best_str = 0;
        int best_j = -1;
        for(int j=0;j<NMarks;j++) {
            if(mk[j].str < 0.1f) continue;
            float d = dist(ag[i].x,ag[i].y,mk[j].x,mk[j].y);
            if(d < 60.0f && mk[j].str > best_str) {
                best_str = mk[j].str; best_j = j;
            }
        }
        if(best_j >= 0) {
            tx = mk[best_j].x; ty = mk[best_j].y;
        }
    }
    
    // Death avoidance: flee from death locations
    if(use_death) {
        for(int j=0;j<mn(*ndeath,100);j++) {
            float dx = (float)death_x[j] - ag[i].x;
            float dy = (float)death_y[j] - ag[i].y;
            if(abs(dx)<30 && abs(dy)<30) {
                float d = sqrtf(dx*dx+dy*dy)+0.01f;
                tx = ag[i].x - dx/d*10.0f;
                ty = ag[i].y - dy/d*10.0f;
                break; // flee from nearest death
            }
        }
    }
    
    // If food nearby, go to cluster center
    if(nearby_food >= 2) {
        tx = cx/nearby_food; ty = cy/nearby_food;
    }
    
    // Move toward target
    float dx=tx-ag[i].x, dy=ty-ag[i].y;
    float d=sqrtf(dx*dx+dy*dy)+0.01f;
    float speed = 2.5f;
    ag[i].x = wrap(ag[i].x + dx/d*speed);
    ag[i].y = wrap(ag[i].y + dy/d*speed);
    
    // Grab nearest food
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        if(dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y) < grab) {
            fd[j].active = 0;
            ag[i].energy += 5.0f;
            break;
        }
    }
    
    // Stigmergy: mark high-density food zones (not eaten food)
    if(use_stig && nearby_food >= 3) {
        // Found a cluster! Mark it
        int si = atomicAdd((int*)&mk[0].x, 1) % NMarks; // use str decay not x
        // Actually use a simple hash
        int si2 = ((int)(cx/nearby_food) * 7 + (int)(cy/nearby_food) * 13) % NMarks;
        mk[si2].x = cx/nearby_food;
        mk[si2].y = cy/nearby_food;
        mk[si2].str = mn((float)nearby_food * 2.0f, 20.0f); // strength = food density
    }
    
    // Energy drain
    ag[i].energy -= 0.015f;
    if(ag[i].energy <= 0) {
        ag[i].alive = 0;
        if(use_death && *ndeath < 100) {
            int di = atomicAdd(ndeath, 1);
            death_x[di] = (int)ag[i].x;
            death_y[di] = (int)ag[i].y;
        }
    }
}

__global__ void decay_marks(Mark *mk) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NMarks) return;
    mk[i].str *= 0.997f;
}

__global__ void respawn(Food *fd, int seed) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD) return;
    if(!fd[i].active && cr2((int*)&seed+i)<0.002f) {
        int s=seed+i*3;
        fd[i].x=cr2(&s)*(float)N; fd[i].y=cr2(&s)*(float)N; fd[i].active=1;
    }
}

__global__ void tally(Agent *ag, float *score, int *alive) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=AGENTS) return;
    if(ag[i].alive) { atomicAdd(score,ag[i].energy); atomicAdd(alive,1); }
}

void run(const char *label, int use_stig, int use_death, int seed) {
    Agent *d_ag; Food *d_fd; Mark *d_mk;
    int *d_dx, *d_dy, *d_nd;
    cudaMalloc(&d_ag, AGENTS*sizeof(Agent));
    cudaMalloc(&d_fd, FOOD*sizeof(Food));
    cudaMalloc(&d_mk, NMarks*sizeof(Mark));
    cudaMalloc(&d_dx, 100*sizeof(int)); cudaMalloc(&d_dy, 100*sizeof(int));
    cudaMalloc(&d_nd, sizeof(int));
    int maxn = 500;
    init_all<<<(maxn+BLK-1)/BLK,BLK>>>(d_ag, d_fd, d_mk, seed);
    cudaMemset(d_nd, 0, sizeof(int));
    
    int nb=(AGENTS+BLK-1)/BLK, nbf=(FOOD+BLK-1)/BLK, nbm=(NMarks+BLK-1)/BLK;
    
    for(int t=0;t<STEPS;t++) {
        step<<<nb,BLK>>>(d_ag,d_fd,d_mk,use_stig,use_death,2.0f,d_dx,d_dy,d_nd,seed+t);
        respawn<<<nbf,BLK>>>(d_fd, seed+t*FOOD);
        decay_marks<<<nbm,BLK>>>(d_mk);
    }
    
    float score=0; int alive=0;
    float *d_s; int *d_a;
    cudaMalloc(&d_s,sizeof(float)); cudaMalloc(&d_a,sizeof(int));
    cudaMemset(d_s,0,sizeof(float)); cudaMemset(d_a,0,sizeof(int));
    tally<<<nb,BLK>>>(d_ag,d_s,d_a);
    cudaMemcpy(&score,d_s,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&alive,d_a,sizeof(int),cudaMemcpyDeviceToHost);
    printf("  %s: score=%.0f alive=%d/256\n", label, score, alive);
    
    cudaFree(d_ag); cudaFree(d_fd); cudaFree(d_mk);
    cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_nd);
    cudaFree(d_s); cudaFree(d_a);
}

int main() {
    printf("=== Stigmergy + Death Observation Synergy (v2) ===\n");
    printf("Agents: 256, Food: 300, Steps: 3000, Grid: 256x256\n\n");
    for(int t=0;t<5;t++) {
        int seed=42+t*1000;
        printf("Trial %d:\n",t+1);
        run("baseline    ", 0,0, seed);
        run("stig-only   ", 1,0, seed);
        run("death-only  ", 0,1, seed);
        run("stig+death  ", 1,1, seed);
        printf("\n");
    }
    return 0;
}
