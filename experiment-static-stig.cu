// Experiment: Stigmergy with Static Food Patches
// Law 190: Does stigmergy help when food is static (no respawn)?
// Hypothesis: Yes — marks point to real food, not stale locations
// Framework: Food collection, 256x256, 256 agents, food patches (clusters)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 256
#define AGENTS 64
#define FOOD 400
#define PATCHES 20        // 20 food patches
#define FOOD_PER_PATCH 20 // 20 food per patch = 400 total
#define STEPS 2000
#define BLK 128
#define NMarks 500

typedef struct { float x,y; int alive; float energy; } Agent;
typedef struct { float x,y; int active; } Food;
typedef struct { float x,y; float str; } Mark;

__device__ static float mn(float a,float b) { return a<b?a:b; }
__device__ static float cr2(int *seed) { *seed = (*seed * 1103515245 + 12345) & 0x7fffffff; return (float)*seed / 0x7fffffff; }
__device__ static float wrap(float v) { return v - floorf(v/(float)N)*(float)N; }
__device__ static float dist(float x1,float y1,float x2,float y2) {
    float dx=abs(x1-x2); float dy=abs(y1-y2);
    dx=mn(dx,N-dx); dy=mn(dy,N-dy); return sqrtf(dx*dx+dy*dy);
}

__global__ void init(Agent *ag, Food *fd, Mark *mk, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<AGENTS) {
        int s=seed+i*137;
        ag[i].x=cr2(&s)*(float)N; ag[i].y=cr2(&s)*(float)N;
        ag[i].alive=1; ag[i].energy=30.0f;
    }
    if(i<NMarks) { mk[i].str=0; mk[i].x=0; mk[i].y=0; }
}

__global__ void init_food_patches(Food *fd, int seed) {
    // Each thread creates one patch center, then places FOOD_PER_PATCH items around it
    int p = blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=PATCHES) return;
    int s = seed+p*997;
    float cx = cr2(&s)*(float)N;
    float cy = cr2(&s)*(float)N;
    for(int j=0;j<FOOD_PER_PATCH;j++) {
        int idx = p*FOOD_PER_PATCH+j;
        if(idx>=FOOD) break;
        int s2 = s+j*31;
        fd[idx].x = wrap(cx + (cr2(&s2)-0.5f)*30.0f);
        fd[idx].y = wrap(cy + (cr2(&s2)-0.5f)*30.0f);
        fd[idx].active = 1;
    }
}

__global__ void step(Agent *ag, Food *fd, Mark *mk, int use_stig, float grab, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=AGENTS || !ag[i].alive) return;
    
    int s=seed+i*31;
    float r1=cr2(&s), r2=cr2(&s);
    
    // Base: random walk
    float tx = ag[i].x + (r1-0.5f)*10.0f;
    float ty = ag[i].y + (r2-0.5f)*10.0f;
    
    // Stigmergy: navigate to marked zones
    if(use_stig) {
        float best_str=0;
        int best_j=-1;
        for(int j=0;j<NMarks;j++) {
            if(mk[j].str < 0.1f) continue;
            float d = dist(ag[i].x,ag[i].y,mk[j].x,mk[j].y);
            if(d < 80.0f && mk[j].str > best_str) {
                best_str=mk[j].str; best_j=j;
            }
        }
        if(best_j>=0) { tx=mk[best_j].x; ty=mk[best_j].y; }
    }
    
    // Scan for nearby food (perception)
    float best_d=FLT_MAX; int best_f=-1;
    int nearby=0; float cx=0,cy=0;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        float d = dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y);
        if(d < grab*1.5f) {
            nearby++; cx+=fd[j].x; cy+=fd[j].y;
            if(d<best_d) { best_d=d; best_f=j; }
        }
    }
    
    // If food detected, go to nearest
    if(best_f>=0) { tx=fd[best_f].x; ty=fd[best_f].y; }
    // If cluster detected, go to center
    else if(nearby>=2) { tx=cx/nearby; ty=cy/nearby; }
    
    // Move
    float dx=tx-ag[i].x, dy=ty-ag[i].y;
    float d=sqrtf(dx*dx+dy*dy)+0.01f;
    ag[i].x = wrap(ag[i].x + dx/d*2.0f);
    ag[i].y = wrap(ag[i].y + dy/d*2.0f);
    
    // Grab food
    int grabbed = 0;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        if(dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y) < grab) {
            fd[j].active = 0;
            ag[i].energy += 5.0f;
            grabbed++;
        }
    }
    
    // Leave stigmergy mark where food was found
    if(use_stig && grabbed > 0) {
        int si = ((int)(ag[i].x)*7 + (int)(ag[i].y)*13) % NMarks;
        mk[si].x = ag[i].x;
        mk[si].y = ag[i].y;
        // Mark strength proportional to food found
        mk[si].str = mn(mk[si].str + (float)grabbed*3.0f, 20.0f);
    }
    
    ag[i].energy -= 0.01f;
    if(ag[i].energy<=0) ag[i].alive=0;
}

__global__ void decay(Mark *mk) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NMarks) return;
    mk[i].str *= 0.999f; // slow decay — food is static, marks stay useful
}

__global__ void tally(Agent *ag, float *score, int *alive, int *food_left, Food *fd) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<AGENTS && ag[i].alive) {
        atomicAdd(score,ag[i].energy);
        atomicAdd(alive,1);
    }
    if(i<FOOD && fd[i].active) atomicAdd(food_left,1);
}

void run(const char *label, int use_stig, int seed) {
    Agent *d_ag; Food *d_fd; Mark *d_mk;
    cudaMalloc(&d_ag,AGENTS*sizeof(Agent));
    cudaMalloc(&d_fd,FOOD*sizeof(Food));
    cudaMalloc(&d_mk,NMarks*sizeof(Mark));
    
    int nb=(AGENTS+BLK-1)/BLK;
    int nbf=(FOOD+BLK-1)/BLK;
    int nbm=(NMarks+BLK-1)/BLK;
    int nbp=(PATCHES+BLK-1)/BLK;
    
    init<<<nb,BLK>>>(d_ag,d_fd,d_mk,seed);
    cudaMemset(d_mk,0,NMarks*sizeof(Mark));
    init_food_patches<<<nbp,BLK>>>(d_fd,seed+999);
    
    for(int t=0;t<STEPS;t++) {
        step<<<nb,BLK>>>(d_ag,d_fd,d_mk,use_stig,2.0f,seed+t);
        decay<<<nbm,BLK>>>(d_mk);
    }
    
    float score=0; int alive=0, fl=0;
    float *d_s; int *d_a, *d_fl;
    cudaMalloc(&d_s,sizeof(float)); cudaMalloc(&d_a,sizeof(int)); cudaMalloc(&d_fl,sizeof(int));
    cudaMemset(d_s,0,sizeof(float)); cudaMemset(d_a,0,sizeof(int)); cudaMemset(d_fl,0,sizeof(int));
    tally<<<(AGENTS+BLK-1)/BLK,BLK>>>(d_ag,d_s,d_a,d_fl,d_fd);
    cudaMemcpy(&score,d_s,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&alive,d_a,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&fl,d_fl,sizeof(int),cudaMemcpyDeviceToHost);
    printf("  %s: score=%.0f alive=%d/256 food_left=%d/%d\n", label, score, alive, fl, FOOD);
    
    cudaFree(d_ag); cudaFree(d_fd); cudaFree(d_mk);
    cudaFree(d_s); cudaFree(d_a); cudaFree(d_fl);
}

int main() {
    printf("=== Stigmergy with Static Food Patches ===\n");
    printf("Agents: 256, Food: 400 (20 patches x 20), No respawn\n");
    printf("Steps: 2000, Grid: 256x256\n\n");
    
    for(int t=0;t<5;t++) {
        int seed=42+t*1000;
        printf("Trial %d:\n",t+1);
        run("baseline    ", 0, seed);
        run("stigmergy   ", 1, seed);
        printf("\n");
    }
    return 0;
}
