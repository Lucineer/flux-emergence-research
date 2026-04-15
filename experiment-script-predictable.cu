#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NModes 2
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        float dist=sqrtf(dx*dx+dy*dy);energy-=0.005f+dist*0.003f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        for(int i=0;i<food_count;i++){
            if(!falive[i])continue;
            float fdx=fx[i]-x,fdy=fy[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            if(fdx*fdx+fdy*fdy<16.0f){int old=atomicExch(&falive[i],0);if(old){energy=fminf(energy+10.0f,200.0f);score+=1.0f;}}
        }
    }
    scores[tid]=score;alive[tid]=(energy>0)?1:0;
}

int main(){
    printf("=== SCRIPT x FOOD PATTERN: Law 245 ===\nN=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    int blk=(N+BLK-1)/BLK;
    // Random food
    float hfx_r[FOOD],hfy_r[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx_r[i]=((float)rand()/RAND_MAX)*W;hfy_r[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx_r,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy_r,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    float tr=0;
    for(int t=0;t<TRIALS;t++){
        cudaMemset(d_fa,1,FOOD*sizeof(int));
        simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,(unsigned int)(42+t*1111));
        cudaDeviceSynchronize();
        float hs[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
        float avg=0;for(int i=0;i<N;i++)avg+=hs[i];tr+=avg/N;
    }
    tr/=TRIALS;
    // Grid food (20x20)
    float hfx_g[FOOD],hfy_g[FOOD];
    int gs=20; // 20x20 = 400
    float spacing=W/gs;
    for(int r=0;r<gs;r++)for(int c=0;c<gs;c++){
        hfx_g[r*gs+c]=(c+0.5f)*spacing;
        hfy_g[r*gs+c]=(r+0.5f)*spacing;
    }
    cudaMemcpy(d_fx,hfx_g,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy_g,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    float tg=0;
    for(int t=0;t<TRIALS;t++){
        cudaMemset(d_fa,1,FOOD*sizeof(int));
        simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,(unsigned int)(42+t*1111));
        cudaDeviceSynchronize();
        float hs[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
        float avg=0;for(int i=0;i<N;i++)avg+=hs[i];tg+=avg/N;
    }
    tg/=TRIALS;
    // Line food
    float hfx_l[FOOD],hfy_l[FOOD];
    for(int i=0;i<FOOD;i++){hfx_l[i]=W/(float)FOOD*(i+0.5f);hfy_l[i]=W/2.0f;}
    cudaMemcpy(d_fx,hfx_l,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy_l,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    float tl=0;
    for(int t=0;t<TRIALS;t++){
        cudaMemset(d_fa,1,FOOD*sizeof(int));
        simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,(unsigned int)(42+t*1111));
        cudaDeviceSynchronize();
        float hs[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
        float avg=0;for(int i=0;i<N;i++)avg+=hs[i];tl+=avg/N;
    }
    tl/=TRIALS;
    printf("Random: score=%.3f\n",tr);
    printf("Grid:   score=%.3f (%+.1f%%)\n",tg,(tg-tr)/tr*100);
    printf("Line:   score=%.3f (%+.1f%%)\n",tl,(tl-tr)/tr*100);
    printf("\n>> Law 245: Does predictable food help scripts?\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}