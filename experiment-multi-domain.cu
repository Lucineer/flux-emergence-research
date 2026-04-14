// experiment-multi-domain.cu — Multi-domain fitness
// From SuperInstance/outcome-tracker: test domain specialization
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 256
#define FOOD 200
#define STEPS 3000
#define WORLD 1024
#define BLOCK 256

__device__ float dax[AGENTS], day[AGENTS], a_energy[AGENTS];
__device__ int a_seed[AGENTS], a_domain[AGENTS];
__device__ float a_score[AGENTS*4];
__device__ float fx[FOOD], fy[FOOD];
__device__ int f_seed[FOOD], f_type[FOOD];
__device__ int d_n, d_nf;

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a,float b) { float d=fabsf(a-b); return fminf(d,WORLD-d); }
__device__ float wr(float v) { float r=fmodf(v,WORLD); return r<0?r+WORLD:r; }

__global__ void init(int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    a_seed[i] = seed+i*137;
    dax[i] = cr(&a_seed[i])*WORLD;
    day[i] = cr(&a_seed[i])*WORLD;
    a_energy[i] = 100.0f;
    a_domain[i] = i % 4;
    for (int d=0;d<4;d++) a_score[i*4+d] = 0;
    if (i < d_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD;
        fy[i] = cr(&f_seed[i])*WORLD;
        f_type[i] = i % 4;
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n || a_energy[i] <= 0) return;
    
    float best_d=1e9, bfx=0, bfy=0;
    for (int j=0;j<d_nf;j++) {
        float dx=wd(dax[i],fx[j]),dy=wd(day[i],fy[j]);
        float d=dx*dx+dy*dy;
        float priority = (f_type[j]==a_domain[i]) ? 0.5f : 1.0f;
        d *= priority;
        if(d<best_d){best_d=d;bfx=fx[j];bfy=fy[j];}
    }
    
    float mx=0,my=0,grab=15.0f,perc=50.0f,speed=3.0f;
    if (best_d < perc*perc) {
        float dx=wd(bfx,dax[i]),dy=wd(bfy,day[i]);
        float d=sqrtf(best_d); if(d>0){mx=dx/d*speed;my=dy/d*speed;}
    } else {
        float ang=cr(&a_seed[i])*6.2832f;
        mx=cosf(ang)*speed; my=sinf(ang)*speed;
    }
    dax[i]=wr(dax[i]+mx); day[i]=wr(day[i]+my);
    a_energy[i] -= 0.02f*speed;
    
    for (int j=0;j<d_nf;j++) {
        float dx=wd(dax[i],fx[j]),dy=wd(day[i],fy[j]);
        if (dx*dx+dy*dy < grab*grab) {
            int t = f_type[j];
            float gain = (t==a_domain[i]) ? 2.0f : 1.0f;
            a_score[i*4+t] += gain;
            a_energy[i] += 5.0f;
            if (a_energy[i]>100.0f) a_energy[i]=100.0f;
            fx[j]=cr(&f_seed[j])*WORLD; fy[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

void run(const char* label, int seed) {
    int _n=AGENTS, _nf=FOOD;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_score[AGENTS*4], h_energy[AGENTS];
    cudaMemcpyFromSymbol(h_score, a_score, sizeof(float)*AGENTS*4);
    cudaMemcpyFromSymbol(h_energy, a_energy, sizeof(float)*AGENTS);
    
    float total=0, alive=0;
    float dt[4]={0};
    for (int i=0;i<AGENTS;i++) {
        if (h_energy[i]<=0) continue;
        alive++;
        for (int d=0;d<4;d++) { total+=h_score[i*4+d]; dt[d]+=h_score[i*4+d]; }
    }
    printf("  %-50s total=%.0f  alive=%d  per=%.1f\n", label, total, alive, alive>0?total/alive:0);
    printf("    combat=%.0f social=%.0f explore=%.0f resource=%.0f\n", dt[0],dt[1],dt[2],dt[3]);
}

int main() {
    printf("=== Multi-Domain Fitness (SuperInstance/outcome-tracker) ===\n\n");
    run("4 domains, specialists (i%%4)", 42);
    run("4 domains, specialists (v2)", 99);
    printf("\nIf domain scores equal: generalization. If skewed: specialization works.\n");
    return 0;
}
