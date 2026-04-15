// Experiment: Stigmergy Accuracy — Mark Placement Strategy
// Law 190: Does WHERE you place the mark matter?
// Compares: mark-at-self (naive) vs mark-at-cluster-center vs mark-at-nearest-food
// Framework: 64 agents, 400 food in 20 patches, limited perception

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 256
#define AGENTS 64
#define FOOD 400
#define PATCHES 20
#define FPP 20
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
    if(i<NMarks) { mk[i].str=0; }
}

__global__ void init_patches(Food *fd, int seed) {
    int p = blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=PATCHES) return;
    int s=seed+p*997;
    float cx=cr2(&s)*(float)N, cy=cr2(&s)*(float)N;
    for(int j=0;j<FPP;j++) {
        int idx=p*FPP+j;
        if(idx>=FOOD) break;
        int s2=s+j*31;
        fd[idx].x=wrap(cx+(cr2(&s2)-0.5f)*30.0f);
        fd[idx].y=wrap(cy+(cr2(&s2)-0.5f)*30.0f);
        fd[idx].active=1;
    }
}

// mark_mode: 0=none, 1=self, 2=cluster_center, 3=nearest_food
__global__ void step(Agent *ag, Food *fd, Mark *mk, int mark_mode, float grab, float perceive, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=AGENTS || !ag[i].alive) return;
    
    int s=seed+i*31;
    float r1=cr2(&s), r2=cr2(&s);
    
    float tx = ag[i].x + (r1-0.5f)*10.0f;
    float ty = ag[i].y + (r2-0.5f)*10.0f;
    
    // Stigmergy navigation
    if(mark_mode > 0) {
        float best_str=0; int best_j=-1;
        for(int j=0;j<NMarks;j++) {
            if(mk[j].str < 0.1f) continue;
            float d = dist(ag[i].x,ag[i].y,mk[j].x,mk[j].y);
            if(d < 80.0f && mk[j].str > best_str) {
                best_str=mk[j].str; best_j=j;
            }
        }
        if(best_j>=0) { tx=mk[best_j].x; ty=mk[best_j].y; }
    }
    
    // Perceive nearby food
    float best_d=FLT_MAX; int best_f=-1;
    int nearby=0; float cx=0,cy=0;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        float d = dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y);
        if(d < perceive) {
            nearby++; cx+=fd[j].x; cy+=fd[j].y;
            if(d<best_d) { best_d=d; best_f=j; }
        }
    }
    
    if(best_f>=0) { tx=fd[best_f].x; ty=fd[best_f].y; }
    else if(nearby>=2) { tx=cx/nearby; ty=cy/nearby; }
    
    // Move
    float dx=tx-ag[i].x, dy=ty-ag[i].y;
    float d=sqrtf(dx*dx+dy*dy)+0.01f;
    ag[i].x = wrap(ag[i].x + dx/d*2.0f);
    ag[i].y = wrap(ag[i].y + dy/d*2.0f);
    
    // Grab
    int grabbed=0; float grab_cx=ag[i].x, grab_cy=ag[i].y;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        if(dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y) < grab) {
            fd[j].active=0; ag[i].energy+=5.0f; grabbed++;
            grab_cx+=fd[j].x; grab_cy+=fd[j].y;
        }
    }
    if(grabbed>0) { grab_cx/=(float)(grabbed+1); grab_cy/=(float)(grabbed+1); }
    
    // Mark placement strategy
    if(mark_mode > 0 && grabbed > 0) {
        int si = ((int)(ag[i].x)*7 + (int)(ag[i].y)*13) % NMarks;
        if(mark_mode == 1) {
            // Mode 1: mark at self position (naive)
            mk[si].x = ag[i].x; mk[si].y = ag[i].y;
        } else if(mark_mode == 2) {
            // Mode 2: mark at cluster center of perceived food
            mk[si].x = cx/nearby; mk[si].y = cy/nearby;
        } else {
            // Mode 3: mark at average grab position
            mk[si].x = grab_cx; mk[si].y = grab_cy;
        }
        mk[si].str = mn(mk[si].str + (float)grabbed*2.0f, 15.0f);
    }
    
    ag[i].energy -= 0.01f;
    if(ag[i].energy<=0) ag[i].alive=0;
}

__global__ void decay(Mark *mk) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NMarks) return;
    mk[i].str *= 0.999f;
}

__global__ void tally(Agent *ag, float *score, int *alive, int *fl, Food *fd) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<AGENTS && ag[i].alive) { atomicAdd(score,ag[i].energy); atomicAdd(alive,1); }
    if(i<FOOD && fd[i].active) atomicAdd(fl,1);
}

void run(const char *label, int mode, int seed) {
    Agent *d_ag; Food *d_fd; Mark *d_mk;
    cudaMalloc(&d_ag,AGENTS*sizeof(Agent));
    cudaMalloc(&d_fd,FOOD*sizeof(Food));
    cudaMalloc(&d_mk,NMarks*sizeof(Mark));
    
    int nb=(AGENTS+BLK-1)/BLK, nbm=(NMarks+BLK-1)/BLK, nbp=(PATCHES+BLK-1)/BLK;
    init<<<nb,BLK>>>(d_ag,d_fd,d_mk,seed);
    cudaMemset(d_mk,0,NMarks*sizeof(Mark));
    init_patches<<<nbp,BLK>>>(d_fd,seed+999);
    
    for(int t=0;t<STEPS;t++) {
        step<<<nb,BLK>>>(d_ag,d_fd,d_mk,mode,2.0f,4.0f,seed+t);
        decay<<<nbm,BLK>>>(d_mk);
    }
    
    float score=0; int alive=0,fl=0;
    float *d_s; int *d_a,*d_fl;
    cudaMalloc(&d_s,sizeof(float)); cudaMalloc(&d_a,sizeof(int)); cudaMalloc(&d_fl,sizeof(int));
    cudaMemset(d_s,0,sizeof(float)); cudaMemset(d_a,0,sizeof(int)); cudaMemset(d_fl,0,sizeof(int));
    tally<<<nb,BLK>>>(d_ag,d_s,d_a,d_fl,d_fd);
    cudaMemcpy(&score,d_s,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&alive,d_a,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&fl,d_fl,sizeof(int),cudaMemcpyDeviceToHost);
    printf("  %-16s: score=%.0f alive=%d food_left=%d\n", label, score, alive, fl);
    
    cudaFree(d_ag); cudaFree(d_fd); cudaFree(d_mk);
    cudaFree(d_s); cudaFree(d_a); cudaFree(d_fl);
}

int main() {
    printf("=== Stigmergy Mark Placement Strategy ===\n");
    printf("64 agents, 400 food (20 patches), perceive=4.0, grab=2.0\n");
    printf("Modes: none, mark-self, mark-cluster-center, mark-grab-center\n\n");
    
    for(int t=0;t<5;t++) {
        int seed=42+t*1000;
        printf("Trial %d:\n",t+1);
        run("no-stigmergy", 0, seed);
        run("mark-self", 1, seed);
        run("mark-cluster", 2, seed);
        run("mark-grab", 3, seed);
        printf("\n");
    }
    return 0;
}
