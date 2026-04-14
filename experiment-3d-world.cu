// experiment-3d-world.cu — 3D toroidal world
// From SuperInstance/voxel-logic: test if 3D changes the laws
// 2D toroidal → 3D toroidal (x,y,z). Same food count, more space.
// Hypothesis: 3D dilutes agent density, perception matters more
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define AGENTS 256
#define FOOD 200
#define STEPS 2500
#define WORLD 128  // 128^3 = 2M volume (vs 1024^2 = 1M in 2D)
#define BLOCK 256

__device__ float ax[AGENTS], ay[AGENTS], az[AGENTS], a_food[AGENTS];
__device__ int a_seed[AGENTS];
__device__ float fx[FOOD], fy[FOOD], fz[FOOD];
__device__ int f_seed[FOOD];
__device__ int d_n, d_nf, d_is_3d;
__device__ float d_grab, d_speed, d_perc;

__device__ float cr(int* s) { *s=(*s*1103515245+12345)&0x7fffffff; return(float)*s/0x7fffffff; }
__device__ float wd(float a,float b,float W) { float d=fabsf(a-b); return fminf(d,W-d); }
__device__ float wr(float v,float W) { float r=fmodf(v,W); return r<0?r+W:r; }

__global__ void init(int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    a_seed[i] = seed+i*137;
    ax[i] = cr(&a_seed[i])*WORLD;
    ay[i] = cr(&a_seed[i])*WORLD;
    az[i] = d_is_3d ? cr(&a_seed[i])*WORLD : 0;
    a_food[i] = 0;
    if (i < d_nf) {
        f_seed[i] = seed+50000+i*997;
        fx[i] = cr(&f_seed[i])*WORLD;
        fy[i] = cr(&f_seed[i])*WORLD;
        fz[i] = d_is_3d ? cr(&f_seed[i])*WORLD : 0;
    }
}

__global__ void step() {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= d_n) return;
    
    float best_d=1e9,bfx=0,bfy=0,bfz=0;
    for (int j=0;j<d_nf;j++) {
        float dx=wd(ax[i],fx[j],WORLD), dy=wd(ay[i],fy[j],WORLD);
        float d=dx*dx+dy*dy;
        if (d_is_3d) { float dz=wd(az[i],fz[j],WORLD); d+=dz*dz; }
        if(d<best_d){best_d=d;bfx=fx[j];bfy=fy[j];bfz=fz[j];}
    }
    
    float mx=0,my=0,mz=0;
    if (best_d < d_perc*d_perc) {
        float dx=wd(bfx,ax[i],WORLD),dy=wd(bfy,ay[i],WORLD);
        float d=sqrtf(best_d);
        if(d>0){mx=dx/d*d_speed;my=dy/d*d_speed;}
        if (d_is_3d) { float dz=wd(bfz,az[i],WORLD); mz=dz/d*d_speed; }
    } else {
        float ang=cr(&a_seed[i])*6.2832f;
        mx=cosf(ang)*d_speed; my=sinf(ang)*d_speed;
        if (d_is_3d) {
            float ang2=cr(&a_seed[i])*6.2832f;
            mz=cosf(ang2)*d_speed*0.5f;
        }
    }
    
    ax[i]=wr(ax[i]+mx,WORLD); ay[i]=wr(ay[i]+my,WORLD);
    if(d_is_3d) az[i]=wr(az[i]+mz,WORLD);
    
    for (int j=0;j<d_nf;j++) {
        float dx=wd(ax[i],fx[j],WORLD),dy=wd(ay[i],fy[j],WORLD);
        float d=dx*dx+dy*dy;
        if (d_is_3d) { float dz=wd(az[i],fz[j],WORLD); d+=dz*dz; }
        if (d < d_grab*d_grab) {
            a_food[i]++;
            fx[j]=cr(&f_seed[j])*WORLD;
            fy[j]=cr(&f_seed[j])*WORLD;
            if(d_is_3d) fz[j]=cr(&f_seed[j])*WORLD;
            break;
        }
    }
}

void run(const char* label, int is_3d, float grab, float perc, float speed, int seed) {
    int _n=AGENTS, _nf=FOOD;
    cudaMemcpyToSymbol(d_n, &_n, sizeof(int));
    cudaMemcpyToSymbol(d_nf, &_nf, sizeof(int));
    cudaMemcpyToSymbol(d_grab, &grab, sizeof(float));
    cudaMemcpyToSymbol(d_speed, &speed, sizeof(float));
    cudaMemcpyToSymbol(d_perc, &perc, sizeof(float));
    cudaMemcpyToSymbol(d_is_3d, &is_3d, sizeof(int));
    
    int blocks=(AGENTS+BLOCK-1)/BLOCK, fblocks=(FOOD+BLOCK-1)/BLOCK;
    init<<<max(blocks,fblocks),BLOCK>>>(seed);
    cudaDeviceSynchronize();
    for (int s=0;s<STEPS;s++) { step<<<blocks,BLOCK>>>(); cudaDeviceSynchronize(); }
    
    float h_food[AGENTS];
    cudaMemcpyFromSymbol(h_food, a_food, sizeof(float)*AGENTS);
    float total=0;
    for (int i=0;i<AGENTS;i++) total+=h_food[i];
    printf("  %-55s total=%.0f  per=%.1f\n", label, total, total/AGENTS);
}

int main() {
    printf("=== 3D World (SuperInstance/voxel-logic) ===\n");
    printf("2D: 1024x1024=1M, 3D: 128^3=2M volume\n\n");
    
    // 2D baseline (world=128, no Z axis)
    printf("--- 2D baselines (128x128) ---\n");
    run("2D, grab=2, perc=8", 0, 2.0f, 8.0f, 1.0f, 42);
    run("2D, grab=4, perc=15", 0, 4.0f, 15.0f, 1.5f, 42);
    run("2D, grab=6, perc=25", 0, 6.0f, 25.0f, 2.0f, 42);
    run("2D, grab=10, perc=40", 0, 10.0f, 40.0f, 3.0f, 42);
    
    // 3D same params
    printf("\n--- 3D (128^3) ---\n");
    run("3D, grab=2, perc=8", 1, 2.0f, 8.0f, 1.0f, 42);
    run("3D, grab=4, perc=15", 1, 4.0f, 15.0f, 1.5f, 42);
    run("3D, grab=6, perc=25", 1, 6.0f, 25.0f, 2.0f, 42);
    run("3D, grab=10, perc=40", 1, 10.0f, 40.0f, 3.0f, 42);
    
    // Perception sweep (key law: perception cliff)
    printf("\n--- 3D perception sweep (grab=5) ---\n");
    run("3D, perc=5", 1, 5.0f, 5.0f, 2.0f, 42);
    run("3D, perc=10", 1, 5.0f, 10.0f, 2.0f, 42);
    run("3D, perc=15", 1, 5.0f, 15.0f, 2.0f, 42);
    run("3D, perc=20", 1, 5.0f, 20.0f, 2.0f, 42);
    run("3D, perc=30", 1, 5.0f, 30.0f, 2.0f, 42);
    run("3D, perc=50", 1, 5.0f, 50.0f, 2.0f, 42);
    
    printf("\n--- 3D vs 2D at equivalent densities ---\n");
    // 2D 1024^2 with 200 food vs 3D 128^3 with 200 food
    // Density 2D: 200/1M = 0.0002. 3D: 200/2M = 0.0001 (half)
    // Fair comparison: 3D with 400 food
    run("3D, grab=5, perc=20, 400 food", 1, 5.0f, 20.0f, 2.0f, 42);
    
    return 0;
}
