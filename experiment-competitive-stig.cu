// Experiment: Stigmergy in 2-Fleet Competitive Framework
// Laws 190-191: Does stigmergy help in adversarial fleet competition?
// Hypothesis: Stigmergy-equipped fleet wins >70% even when outnumbered
// Framework: 256x256, two fleets of agents, shared food pool

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 256
#define FLEET_A 128
#define FLEET_B 128
#define FOOD 500
#define STEPS 2000
#define BLK 128

typedef struct { float x,y,vx,vy; int alive; float energy; int fleet; } Agent;
typedef struct { float x,y; int active; } Food;
typedef struct { float x,y; float str; } Mark;

__device__ static float mn(float a,float b) { return a<b?a:b; }
__device__ static float cr2(int *seed) { *seed = (*seed * 1103515245 + 12345) & 0x7fffffff; return (float)*seed / 0x7fffffff; }
__device__ static float wrap(float v) { return v - floorf(v/(float)N)*(float)N; }
__device__ static float dist(float x1,float y1,float x2,float y2) { float dx=abs(x1-x2); float dy=abs(y1-y2); dx=mn(dx,N-dx); dy=mn(dy,N-dy); return sqrtf(dx*dx+dy*dy); }

__global__ void init_agents(Agent *ag, int n, int fleet, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int s = seed+i*137+fleet*9999;
    ag[i].x = cr2(&s)*(float)N;
    ag[i].y = cr2(&s)*(float)N;
    ag[i].vx=0; ag[i].vy=0;
    ag[i].alive=1; ag[i].energy=50.0f; ag[i].fleet=fleet;
}

__global__ void init_food(Food *fd, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD) return;
    int s = seed+i*251;
    fd[i].x = cr2(&s)*(float)N;
    fd[i].y = cr2(&s)*(float)N;
    fd[i].active=1;
}

__global__ void step_fleet(Agent *ag, Food *fd, Mark *marks_a, Mark *marks_b,
                           int n, int other_n, int offset, int other_offset,
                           int use_stig_a, int use_stig_b, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int gi = offset+i;
    if(!ag[gi].alive) return;
    
    int s = seed+i*31+gi*7;
    float r1=cr2(&s), r2=cr2(&s);
    int fleet = ag[gi].fleet;
    int use_stig = fleet==0 ? use_stig_a : use_stig_b;
    Mark *my_marks = fleet==0 ? marks_a : marks_b;
    
    // Default: random walk toward nearest food
    float best_d=FLT_MAX; int best_f=-1;
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        float d = dist(ag[gi].x,ag[gi].y,fd[j].x,fd[j].y);
        if(d<best_d) { best_d=d; best_f=j; }
    }
    
    float tx = ag[gi].x + (r1-0.5f)*8.0f;
    float ty = ag[gi].y + (r2-0.5f)*8.0f;
    
    // Stigmergy: move toward strongest nearby mark
    if(use_stig) {
        float best_sd=FLT_MAX, best_ss=0;
        for(int j=0;j<500;j++) {
            if(my_marks[j].str<0.05f) continue;
            float d = dist(ag[gi].x,ag[gi].y,my_marks[j].x,my_marks[j].y);
            if(d<50.0f && my_marks[j].str>best_ss) {
                best_sd=d; best_ss=my_marks[j].str;
                tx=my_marks[j].x; ty=my_marks[j].y;
            }
        }
    }
    
    // Go to food if found
    if(best_f>=0 && best_d<20.0f) { tx=fd[best_f].x; ty=fd[best_f].y; }
    
    // Move
    float dx=tx-ag[gi].x, dy=ty-ag[gi].y;
    float d=sqrtf(dx*dx+dy*dy)+0.01f;
    ag[gi].x = wrap(ag[gi].x + dx/d*2.0f);
    ag[gi].y = wrap(ag[gi].y + dy/d*2.0f);
    
    // Grab food
    for(int j=0;j<FOOD;j++) {
        if(!fd[j].active) continue;
        if(dist(ag[gi].x,ag[gi].y,fd[j].x,fd[j].y)<2.0f) {
            fd[j].active=0;
            ag[gi].energy += 3.0f;
            // Leave stigmergy mark
            if(use_stig) {
                int si = (i*7+j*3)%500;
                my_marks[si].x = fd[j].x;
                my_marks[si].y = fd[j].y;
                my_marks[si].str = 10.0f;
            }
            break;
        }
    }
    
    ag[gi].energy -= 0.02f;
    if(ag[gi].energy<=0) ag[gi].alive=0;
}

__global__ void respawn(Food *fd, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD) return;
    if(!fd[i].active && cr2((int*)&seed+i)<0.003f) {
        int s=seed+i*3;
        fd[i].x=cr2(&s)*(float)N;
        fd[i].y=cr2(&s)*(float)N;
        fd[i].active=1;
    }
}

__global__ void decay_marks(Mark *m, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    m[i].str*=0.995f;
}

__global__ void tally(Agent *ag, int offset, int n, float *score) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    atomicAdd(score, ag[offset+i].energy);
}

void run(const char *label, int stig_a, int stig_b, int seed) {
    int total = FLEET_A+FLEET_B;
    Agent *d_ag; Food *d_fd; Mark *d_ma, *d_mb;
    cudaMalloc(&d_ag, total*sizeof(Agent));
    cudaMalloc(&d_fd, FOOD*sizeof(Food));
    cudaMalloc(&d_ma, 500*sizeof(Mark)); cudaMalloc(&d_mb, 500*sizeof(Mark));
    cudaMemset(d_ma,0,500*sizeof(Mark)); cudaMemset(d_mb,0,500*sizeof(Mark));
    
    int nb=(total+BLK-1)/BLK, nbf=(FOOD+BLK-1)/BLK, nbm=(500+BLK-1)/BLK;
    
    init_agents<<<(FLEET_A+BLK-1)/BLK,BLK>>>(d_ag, FLEET_A, 0, seed);
    init_agents<<<(FLEET_B+BLK-1)/BLK,BLK>>>(d_ag+FLEET_A, FLEET_B, 1, seed+500);
    init_food<<<nbf,BLK>>>(d_fd, seed+999);
    
    for(int t=0;t<STEPS;t++) {
        step_fleet<<<(FLEET_A+BLK-1)/BLK,BLK>>>(d_ag,d_fd,d_ma,d_mb,
            FLEET_A,FLEET_B,0,FLEET_A,stig_a,stig_b,seed+t);
        step_fleet<<<(FLEET_B+BLK-1)/BLK,BLK>>>(d_ag,d_fd,d_ma,d_mb,
            FLEET_B,FLEET_A,FLEET_A,0,stig_a,stig_b,seed+t+999);
        respawn<<<nbf,BLK>>>(d_fd, seed+t*FOOD);
        decay_marks<<<nbm,BLK>>>(d_ma, 500);
        decay_marks<<<nbm,BLK>>>(d_mb, 500);
    }
    
    float sa=0,sb=0;
    float *d_sa,*d_sb;
    cudaMalloc(&d_sa,sizeof(float)); cudaMalloc(&d_sb,sizeof(float));
    cudaMemset(d_sa,0,sizeof(float)); cudaMemset(d_sb,0,sizeof(float));
    tally<<<(FLEET_A+BLK-1)/BLK,BLK>>>(d_ag,0,FLEET_A,d_sa);
    tally<<<(FLEET_B+BLK-1)/BLK,BLK>>>(d_ag,FLEET_A,FLEET_B,d_sb);
    cudaMemcpy(&sa,d_sa,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&sb,d_sb,sizeof(float),cudaMemcpyDeviceToHost);
    
    float ratio = sb>0.01f ? sa/sb : 99.0f;
    printf("  %s: A=%.0f B=%.0f ratio=%.3f\n", label, sa, sb, ratio);
    
    cudaFree(d_ag); cudaFree(d_fd); cudaFree(d_ma); cudaFree(d_mb);
    cudaFree(d_sa); cudaFree(d_sb);
}

int main() {
    printf("=== Stigmergy in 2-Fleet Competition ===\n");
    printf("Fleet A: 128 agents, Fleet B: 128 agents, Food: 500\n\n");
    
    for(int t=0;t<5;t++) {
        int seed=42+t*1000;
        printf("Trial %d:\n",t+1);
        run("no-stig    ", 0,0, seed);
        run("A-stig     ", 1,0, seed);
        run("B-stig     ", 0,1, seed);
        run("both-stig  ", 1,1, seed);
        printf("\n");
    }
    return 0;
}
