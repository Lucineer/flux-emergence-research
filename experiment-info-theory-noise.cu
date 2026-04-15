#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NF 128
#define FOOD 400
#define W 256
#define BLK 128
#define GRID 64
#define NNOISE 4
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void phase1_traces(int *traces, int steps, int n, int w, int grid_w,
    float noise, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        dx+=(cr(&rng)-0.5f)*2.0f*noise;dy+=(cr(&rng)-0.5f)*2.0f*noise;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        int gx=(int)(x/w*grid_w)%grid_w;
        int gy=(int)(y/w*grid_w)%grid_w;
        atomicAdd(&traces[gy*grid_w+gx],1);
    }
}

__global__ void phase2_exploit(float *scores, int *traces, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, int grid_w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + (tid+n) * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = (tid+n) * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        int gx=(int)(x/w*grid_w)%grid_w;
        int gy=(int)(y/w*grid_w)%grid_w;
        float best_v=0,best_dx=0,best_dy=0;
        for(int dy2=-2;dy2<=2;dy2++)for(int dx2=-2;dx2<=2;dx2++){
            if(dx2==0&&dy2==0)continue;
            int nx=(gx+dx2+grid_w)%grid_w, ny=(gy+dy2+grid_w)%grid_w;
            int v=traces[ny*grid_w+nx];
            if(v>best_v){best_v=v;best_dx=dx2*w/grid_w;best_dy=dy2*w/grid_w;}
        }
        dx+=best_dx*0.1f;dy+=best_dy*0.1f;
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
    scores[tid]=score;
}

int main(){
    float noises[]={0.0f,0.5f,1.0f,2.0f};
    printf("=== NOISE INFORMATION CAPACITY: Law 258 ===\nN=%d Food=%d Steps=%d Trials=%d\n\n",NF,FOOD,1500,TRIALS);
    int *d_tr;cudaMalloc(&d_tr,GRID*GRID*sizeof(int));
    float *d_s,*d_fx,*d_fy;int *d_fa;
    cudaMalloc(&d_s,NF*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(NF+BLK-1)/BLK;
    printf("%-10s | %-8s | %s\n","Noise","Score","vs no-traces");
    printf("----------|----------|----------\n");
    for(int n=0;n<NNOISE;n++){
        float ts=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            cudaMemset(d_tr,0,GRID*GRID*sizeof(int));
            cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            phase1_traces<<<blk,BLK>>>(d_tr,1500,NF,W,GRID,noises[n],(unsigned int)(42+tr*1111));
            cudaDeviceSynchronize();
            phase2_exploit<<<blk,BLK>>>(d_s,d_tr,d_fx,d_fy,d_fa,1500,NF,FOOD,W,GRID,(unsigned int)(42+tr*1111));
            cudaDeviceSynchronize();
            float hs[NF];cudaMemcpy(hs,d_s,NF*sizeof(float),cudaMemcpyDeviceToHost);
            float avg=0;for(int i=0;i<NF;i++)avg+=hs[i];ts+=avg/NF;
        }
        ts/=TRIALS;
        printf("noise=%.1f  | %.3f     | %+.1f%%\n",noises[n],ts,n==0?0.0:(ts-0.881)/0.881*100);
    }
    printf("\n>> Law 258: Information capacity of noise traces\n");
    cudaFree(d_tr);cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);
    return 0;
}