// Experiment: Territory Grid Resolution Sweep
// Law 193: What's the optimal territory grid granularity?
// Hypothesis: Too fine = no signal, too coarse = no discrimination
// Framework: 64 agents, 400 food in 20 patches, territory avoidance

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
#define MAXGRID 64

typedef struct { float x,y; int alive; float energy; } Agent;
typedef struct { float x,y; int active; } Food;

__device__ int territory[MAXGRID*MAXGRID];

__device__ static float mn(float a,float b) { return a<b?a:b; }
__device__ static float cr2(int *seed) { *seed = (*seed * 1103515245 + 12345) & 0x7fffffff; return (float)*seed / 0x7fffffff; }
__device__ static float wrap(float v) { return v - floorf(v/(float)N)*(float)N; }
__device__ static float dist(float x1,float y1,float x2,float y2) {
    float dx=abs(x1-x2); float dy=abs(y1-y2);
    dx=mn(dx,N-dx); dy=mn(dy,N-dy); return sqrtf(dx*dx+dy*dy);
}

__global__ void init(Agent *ag, Food *fd, int ng) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<AGENTS) {
        int s=42+i*137;
        ag[i].x=cr2(&s)*(float)N; ag[i].y=cr2(&s)*(float)N;
        ag[i].alive=1; ag[i].energy=30.0f;
    }
    if(i<ng*ng) territory[i]=0;
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

__global__ void step(Agent *ag, Food *fd, int ng, int use_stig, float grab, float perceive, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=AGENTS || !ag[i].alive) return;
    
    int s=seed+i*31;
    float r1=cr2(&s), r2=cr2(&s);
    
    int gc = (int)(ag[i].x / (float)N * ng) % ng;
    int gr = (int)(ag[i].y / (float)N * ng) % ng;
    atomicAdd(&territory[gr*ng+gc], 1);
    
    float tx = ag[i].x + (r1-0.5f)*12.0f;
    float ty = ag[i].y + (r2-0.5f)*12.0f;
    
    if(use_stig) {
        float best_score = FLT_MAX;
        for(int dr=-2;dr<=2;dr++) {
            for(int dc=-2;dc<=2;dc++) {
                int nr = ((gr+dr)%ng+ng)%ng;
                int nc = ((gc+dc)%ng+ng)%ng;
                if(nr>=ng||nc>=ng) continue;
                float score = (float)territory[nr*ng+nc] + (abs(dr)+abs(dc))*2.0f;
                if(score < best_score) {
                    best_score = score;
                    tx = ((float)nc + 0.5f) / (float)ng * (float)N;
                    ty = ((float)nr + 0.5f) / (float)ng * (float)N;
                }
            }
        }
    }
    
    float best_d=FLT_MAX; int best_f=-1;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        float d = dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y);
        if(d < perceive && d < best_d) { best_d=d; best_f=j; }
    }
    if(best_f>=0) { tx=fd[best_f].x; ty=fd[best_f].y; }
    
    float dx=tx-ag[i].x, dy=ty-ag[i].y;
    float d=sqrtf(dx*dx+dy*dy)+0.01f;
    ag[i].x = wrap(ag[i].x + dx/d*2.0f);
    ag[i].y = wrap(ag[i].y + dy/d*2.0f);
    
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        if(dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y) < grab) {
            fd[j].active=0; ag[i].energy+=5.0f; break;
        }
    }
    
    ag[i].energy -= 0.01f;
    if(ag[i].energy<=0) ag[i].alive=0;
}

__global__ void tally(Agent *ag, float *score, int *alive, int *fl, Food *fd) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<AGENTS && ag[i].alive) { atomicAdd(score,ag[i].energy); atomicAdd(alive,1); }
    if(i<FOOD && fd[i].active) atomicAdd(fl,1);
}

void run(const char *label, int ng, int use_stig, int seed) {
    Agent *d_ag; Food *d_fd;
    cudaMalloc(&d_ag,AGENTS*sizeof(Agent));
    cudaMalloc(&d_fd,FOOD*sizeof(Food));
    
    int nb=(AGENTS+BLK-1)/BLK, nbp=(PATCHES+BLK-1)/BLK;
    
    init<<<(AGENTS+BLK-1)/BLK,BLK>>>(d_ag,d_fd,ng);
    init_patches<<<nbp,BLK>>>(d_fd,seed+999);
    
    for(int t=0;t<STEPS;t++)
        step<<<nb,BLK>>>(d_ag,d_fd,ng,use_stig,2.0f,4.0f,seed+t);
    
    float score=0; int alive=0,fl=0;
    float *d_s; int *d_a,*d_fl;
    cudaMalloc(&d_s,sizeof(float)); cudaMalloc(&d_a,sizeof(int)); cudaMalloc(&d_fl,sizeof(int));
    cudaMemset(d_s,0,sizeof(float)); cudaMemset(d_a,0,sizeof(int)); cudaMemset(d_fl,0,sizeof(int));
    tally<<<nb,BLK>>>(d_ag,d_s,d_a,d_fl,d_fd);
    cudaMemcpy(&score,d_s,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&alive,d_a,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(&fl,d_fl,sizeof(int),cudaMemcpyDeviceToHost);
    printf("  %-24s: score=%.0f food_left=%d\n", label, score, fl);
    
    cudaFree(d_ag); cudaFree(d_fd); cudaFree(d_s); cudaFree(d_a); cudaFree(d_fl);
}

int main() {
    printf("=== Territory Grid Resolution Sweep ===\n");
    printf("64 agents, 400 food (20 patches), territory avoidance\n\n");
    
    int grids[] = {4, 8, 16, 32, 64};
    int ngrids = 5;
    
    for(int t=0;t<3;t++) {
        int seed=42+t*1000;
        printf("Trial %d:\n",t+1);
        run("baseline (no stig)   ", 32, 0, seed);
        for(int g=0;g<ngrids;g++) {
            char label[64];
            snprintf(label, sizeof(label), "territory grid=%d   ", grids[g]);
            run(label, grids[g], 1, seed);
        }
        printf("\n");
    }
    return 0;
}
