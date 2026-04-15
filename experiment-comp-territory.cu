// Experiment: Territory Avoidance in Competitive 2-Fleet
// Law 194: Does negative stigmergy (territory avoid) work in competition?
// Hypothesis: Territory-aware fleet explores more efficiently, wins resource war
// Framework: 2 fleets x 64 agents, 300 food, territory grid

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 256
#define FLEET 64
#define FOOD 300
#define STEPS 2000
#define BLK 128
#define NG 16  // territory grid 16x16

typedef struct { float x,y; int alive; float energy; int fleet; } Agent;
typedef struct { float x,y; int active; } Food;

__device__ int territory_a[NG*NG];
__device__ int territory_b[NG*NG];

__device__ static float mn(float a,float b) { return a<b?a:b; }
__device__ static float cr2(int *seed) { *seed = (*seed * 1103515245 + 12345) & 0x7fffffff; return (float)*seed / 0x7fffffff; }
__device__ static float wrap(float v) { return v - floorf(v/(float)N)*(float)N; }
__device__ static float dist(float x1,float y1,float x2,float y2) {
    float dx=abs(x1-x2); float dy=abs(y1-y2);
    dx=mn(dx,N-dx); dy=mn(dy,N-dy); return sqrtf(dx*dx+dy*dy);
}

__global__ void init(Agent *ag, Food *fd, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int total = FLEET*2;
    if(i<total) {
        int s=seed+i*137;
        ag[i].x=cr2(&s)*(float)N; ag[i].y=cr2(&s)*(float)N;
        ag[i].alive=1; ag[i].energy=30.0f;
        ag[i].fleet = (i<FLEET)?0:1;
    }
    if(i<FOOD) {
        int s=seed+i*251+999;
        fd[i].x=cr2(&s)*(float)N; fd[i].y=cr2(&s)*(float)N;
        fd[i].active=1;
    }
    if(i<NG*NG) { territory_a[i]=0; territory_b[i]=0; }
}

__global__ void respawn(Food *fd, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD) return;
    if(!fd[i].active && cr2((int*)&seed+i)<0.002f) {
        int s=seed+i*3;
        fd[i].x=cr2(&s)*(float)N; fd[i].y=cr2(&s)*(float)N;
        fd[i].active=1;
    }
}

__global__ void step(Agent *ag, Food *fd, int stig_a, int stig_b, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FLEET*2 || !ag[i].alive) return;
    
    int s=seed+i*31;
    float r1=cr2(&s), r2=cr2(&s);
    int fleet = ag[i].fleet;
    int use_stig = fleet==0 ? stig_a : stig_b;
    int *my_territory = fleet==0 ? territory_a : territory_b;
    
    int gc = (int)(ag[i].x / (float)N * NG) % NG;
    int gr = (int)(ag[i].y / (float)N * NG) % NG;
    atomicAdd(&my_territory[gr*NG+gc], 1);
    
    float tx = ag[i].x + (r1-0.5f)*10.0f;
    float ty = ag[i].y + (r2-0.5f)*10.0f;
    
    if(use_stig) {
        float best_score = FLT_MAX;
        for(int dr=-2;dr<=2;dr++) {
            for(int dc=-2;dc<=2;dc++) {
                int nr = ((gr+dr)%NG+NG)%NG;
                int nc = ((gc+dc)%NG+NG)%NG;
                float score = (float)my_territory[nr*NG+nc] + (abs(dr)+abs(dc))*1.5f;
                if(score < best_score) {
                    best_score = score;
                    tx = ((float)nc+0.5f)/(float)NG*(float)N;
                    ty = ((float)nr+0.5f)/(float)NG*(float)N;
                }
            }
        }
    }
    
    // Scan food
    float best_d=FLT_MAX; int best_f=-1;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        float d = dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y);
        if(d < 4.0f && d < best_d) { best_d=d; best_f=j; }
    }
    if(best_f>=0) { tx=fd[best_f].x; ty=fd[best_f].y; }
    
    float dx=tx-ag[i].x, dy=ty-ag[i].y;
    float d=sqrtf(dx*dx+dy*dy)+0.01f;
    ag[i].x = wrap(ag[i].x + dx/d*2.0f);
    ag[i].y = wrap(ag[i].y + dy/d*2.0f);
    
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        if(dist(ag[i].x,ag[i].y,fd[j].x,fd[j].y) < 2.0f) {
            fd[j].active=0; ag[i].energy+=5.0f; break;
        }
    }
    
    ag[i].energy -= 0.015f;
    if(ag[i].energy<=0) ag[i].alive=0;
}

__global__ void tally(Agent *ag, int offset, int n, float *score) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || !ag[offset+i].alive) return;
    atomicAdd(score, ag[offset+i].energy);
}

void run(const char *label, int sa, int sb, int seed) {
    Agent *d_ag; Food *d_fd;
    cudaMalloc(&d_ag, FLEET*2*sizeof(Agent));
    cudaMalloc(&d_fd, FOOD*sizeof(Food));
    
    int total = FLEET*2;
    init<<<(total+BLK-1)/BLK,BLK>>>(d_ag, d_fd, seed);
    
    int nb=(FLEET+BLK-1)/BLK, nbf=(FOOD+BLK-1)/BLK;
    for(int t=0;t<STEPS;t++) {
        step<<<(total+BLK-1)/BLK,BLK>>>(d_ag,d_fd,sa,sb,seed+t);
        respawn<<<nbf,BLK>>>(d_fd,seed+t*FOOD);
    }
    
    float sa_score=0,sb_score=0;
    float *d_sa,*d_sb;
    cudaMalloc(&d_sa,sizeof(float)); cudaMalloc(&d_sb,sizeof(float));
    cudaMemset(d_sa,0,sizeof(float)); cudaMemset(d_sb,0,sizeof(float));
    tally<<<nb,BLK>>>(d_ag,0,FLEET,d_sa);
    tally<<<nb,BLK>>>(d_ag,FLEET,FLEET,d_sb);
    cudaMemcpy(&sa_score,d_sa,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&sb_score,d_sb,sizeof(float),cudaMemcpyDeviceToHost);
    
    float ratio = sb_score>0.01f ? sa_score/sb_score : 99.0f;
    printf("  %-20s: A=%.0f B=%.0f A/B=%.3f total=%.0f\n",
           label, sa_score, sb_score, ratio, sa_score+sb_score);
    
    cudaFree(d_ag); cudaFree(d_fd); cudaFree(d_sa); cudaFree(d_sb);
}

int main() {
    printf("=== Territory Avoidance in 2-Fleet Competition ===\n");
    printf("Fleet A: 64, Fleet B: 64, Food: 300, Grid: 16x16\n\n");
    
    for(int t=0;t<5;t++) {
        int seed=42+t*1000;
        printf("Trial %d:\n",t+1);
        run("no-stig       ", 0,0, seed);
        run("A-stig        ", 1,0, seed);
        run("B-stig        ", 0,1, seed);
        run("both-stig     ", 1,1, seed);
        printf("\n");
    }
    return 0;
}
