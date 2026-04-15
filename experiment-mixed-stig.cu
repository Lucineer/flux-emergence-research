// Experiment: Mixed Fleet — Territory Avoidance Fraction Sweep
// Law 196: What fraction of agents should use territory avoidance in competition?
// Hypothesis: Small fraction avoids (scouts), majority seeks (collectors) = optimal
// Framework: 2 fleets x 64, mixed avoidance fractions

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 256
#define FA 64
#define FB 64
#define FOOD 300
#define STEPS 2000
#define BLK 128
#define NG 16

typedef struct { float x,y; int alive; float energy; int fleet,stig; } Agent;
typedef struct { float x,y; int active; } Food;

__device__ int terr[NG*NG];

__device__ static float mn(float a,float b) { return a<b?a:b; }
__device__ static float cr2(int *seed) { *seed = (*seed * 1103515245 + 12345) & 0x7fffffff; return (float)*seed / 0x7fffffff; }
__device__ static float wrap(float v) { return v - floorf(v/(float)N)*(float)N; }
__device__ static float dist(float x1,float y1,float x2,float y2) {
    float dx=abs(x1-x2); float dy=abs(y1-y2);
    dx=mn(dx,N-dx); dy=mn(dy,N-dy); return sqrtf(dx*dx+dy*dy);
}

__global__ void init(Agent *ag, Food *fd, float stig_frac_a, float stig_frac_b, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int total = FA+FB;
    if(i<total) {
        int s=seed+i*137;
        ag[i].x=cr2(&s)*(float)N; ag[i].y=cr2(&s)*(float)N;
        ag[i].alive=1; ag[i].energy=30.0f;
        ag[i].fleet = (i<FA)?0:1;
        float frac = (i<FA) ? stig_frac_a : stig_frac_b;
        ag[i].stig = (i % (FA) < (int)(frac*(float)FA)) ? 1 : 0;
    }
    if(i<FOOD) {
        int s=seed+i*251+999;
        fd[i].x=cr2(&s)*(float)N; fd[i].y=cr2(&s)*(float)N;
        fd[i].active=1;
    }
    if(i<NG*NG) terr[i]=0;
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

__global__ void step(Agent *ag, Food *fd, int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FA+FB || !ag[i].alive) return;
    
    int s=seed+i*31;
    float r1=cr2(&s), r2=cr2(&s);
    
    int gc=(int)(ag[i].x/(float)N*NG)%NG;
    int gr=(int)(ag[i].y/(float)N*NG)%NG;
    atomicAdd(&terr[gr*NG+gc], 1);
    
    float tx = ag[i].x + (r1-0.5f)*10.0f;
    float ty = ag[i].y + (r2-0.5f)*10.0f;
    
    if(ag[i].stig) {
        float best_score = FLT_MAX;
        for(int dr=-2;dr<=2;dr++) {
            for(int dc=-2;dc<=2;dc++) {
                int nr=((gr+dr)%NG+NG)%NG;
                int nc=((gc+dc)%NG+NG)%NG;
                float score = (float)terr[nr*NG+nc] + (abs(dr)+abs(dc))*1.5f;
                if(score < best_score) {
                    best_score = score;
                    tx = ((float)nc+0.5f)/(float)NG*(float)N;
                    ty = ((float)nr+0.5f)/(float)NG*(float)N;
                }
            }
        }
    }
    
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

void run(const char *label, float frac_a, float frac_b, int seed) {
    Agent *d_ag; Food *d_fd;
    cudaMalloc(&d_ag,(FA+FB)*sizeof(Agent));
    cudaMalloc(&d_fd,FOOD*sizeof(Food));
    
    init<<<((FA+FB)+BLK-1)/BLK,BLK>>>(d_ag,d_fd,frac_a,frac_b,seed);
    
    int nb=(FA+BLK-1)/BLK, nbf=(FOOD+BLK-1)/BLK;
    for(int t=0;t<STEPS;t++) {
        step<<<((FA+FB)+BLK-1)/BLK,BLK>>>(d_ag,d_fd,seed+t);
        respawn<<<nbf,BLK>>>(d_fd,seed+t*FOOD);
    }
    
    float sa=0,sb=0;
    float *d_sa,*d_sb;
    cudaMalloc(&d_sa,sizeof(float)); cudaMalloc(&d_sb,sizeof(float));
    cudaMemset(d_sa,0,sizeof(float)); cudaMemset(d_sb,0,sizeof(float));
    tally<<<nb,BLK>>>(d_ag,0,FA,d_sa);
    tally<<<nb,BLK>>>(d_ag,FA,FB,d_sb);
    cudaMemcpy(&sa,d_sa,sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&sb,d_sb,sizeof(float),cudaMemcpyDeviceToHost);
    
    float ratio = sb>0.01f ? sa/sb : 99.0f;
    printf("  %-28s: A=%.0f B=%.0f A/B=%.2f total=%.0f\n",
           label, sa, sb, ratio, sa+sb);
    
    cudaFree(d_ag); cudaFree(d_fd); cudaFree(d_sa); cudaFree(d_sb);
}

int main() {
    printf("=== Mixed Fleet: Territory Avoidance Fraction Sweep ===\n");
    printf("Fleet A: 64, Fleet B: 64, Food: 300, Grid: 16x16\n");
    printf("A always 0%% stig. B varies 0-100%%.\n\n");
    
    float fracs[] = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f};
    int nfracs = 6;
    
    for(int t=0;t<3;t++) {
        int seed=42+t*1000;
        printf("Trial %d:\n",t+1);
        for(int f=0;f<nfracs;f++) {
            char label[64];
            snprintf(label,sizeof(label),"B-stig=%.0f%%", fracs[f]*100);
            run(label, 0.0f, fracs[f], seed);
        }
        printf("\n");
    }
    return 0;
}
