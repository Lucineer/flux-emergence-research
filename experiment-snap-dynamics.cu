#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define STEPS 10000
#define W 256
#define BLK 128
#define NRUNS 20

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(int *snap_step, int steps, int n, int w,
    float energy_budget, float perc_cost, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = energy_budget;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    int my_snap = steps;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        float dist=sqrtf(dx*dx+dy*dy);energy-=0.005f+dist*0.003f+perc_cost;
        if(energy<=0){my_snap=t;break;}
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
    }
    snap_step[tid]=my_snap;
}

int main(){
    printf("=== SNAP DYNAMICS: Law 266 ===\n\n");
    int *d_snap;cudaMalloc(&d_snap,N*sizeof(int));
    int blk=(N+BLK-1)/BLK;
    // Test 1: Determinism
    printf("TEST 1: Determinism (%d runs, same seed)\n",NRUNS);
    float times[NRUNS];
    for(int r=0;r<NRUNS;r++){
        simulate<<<blk,BLK>>>(d_snap,STEPS,N,W,80.0f,0.0f,42u);
        cudaDeviceSynchronize();
        int hs[N];cudaMemcpy(hs,d_snap,N*sizeof(int),cudaMemcpyDeviceToHost);
        float avg=0;for(int i=0;i<N;i++)avg+=hs[i];times[r]=avg/N;
    }
    float mn=99999,mx=0;
    for(int r=0;r<NRUNS;r++){if(times[r]<mn)mn=times[r];if(times[r]>mx)mx=times[r];}
    printf("  Range: %.1f - %.1f spread: %.6f\n",mn,mx,mx-mn);
    printf("  %s\n\n",mx-mn<0.001?"DETERMINISTIC":"CHAOTIC");
    // Test 2: Energy sensitivity
    printf("TEST 2: Energy cliff sensitivity\n");
    float energies[]={20,30,35,38,39,40,41,42,45,50,60,80,100};
    for(int e=0;e<13;e++){
        simulate<<<blk,BLK>>>(d_snap,STEPS,N,W,energies[e],0.0f,42u);
        cudaDeviceSynchronize();
        int hs[N];cudaMemcpy(hs,d_snap,N*sizeof(int),cudaMemcpyDeviceToHost);
        int surv=0;float avg=0;
        for(int i=0;i<N;i++){if(hs[i]==STEPS)surv++;avg+=hs[i];}
        printf("  E=%.0f: %d/%d survived (%.0f%%) avg_snap=%.1f\n",energies[e],surv,N,surv*100.0/N,avg/N);
    }
    // Test 3: Perception cost sensitivity
    printf("\nTEST 3: Perception cost sensitivity (energy=50)\n");
    float costs[]={0.0f,0.01f,0.02f,0.03f,0.04f,0.05f,0.06f,0.08f,0.10f};
    for(int c=0;c<9;c++){
        simulate<<<blk,BLK>>>(d_snap,STEPS,N,W,50.0f,costs[c],42u);
        cudaDeviceSynchronize();
        int hs[N];cudaMemcpy(hs,d_snap,N*sizeof(int),cudaMemcpyDeviceToHost);
        int surv=0;
        for(int i=0;i<N;i++)if(hs[i]==STEPS)surv++;
        printf("  cost=%.3f: %d/%d (%.0f%%)\n",costs[c],surv,N,surv*100.0/N);
    }
    printf("\n>> Law 266\n");
    cudaFree(d_snap);return 0;
}