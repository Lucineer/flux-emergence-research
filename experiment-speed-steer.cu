#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSPEEDS 5
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, int use_steer, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];
    for(int i=0;i<8;i++) dir[i]=base_angle+i*0.785f;
    float marks_x[10], marks_y[10];
    int nmarks=0;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;
        float dx=cosf(dir[p])*2.0f, dy=sinf(dir[p])*2.0f;
        if(use_steer){
            float sx=0,sy=0;
            for(int m=0;m<nmarks;m++){
                float mx=marks_x[m]-x,my=marks_y[m]-y;
                if(mx>w/2)mx-=w;if(mx<-w/2)mx+=w;
                if(my>w/2)my-=w;if(my<-w/2)my+=w;
                float d=mx*mx+my*my;
                if(d<400.0f && d>0.01f){sx-=mx/sqrtf(d);sy-=my/sqrtf(d);}
            }
            dx+=sx*1.5f; dy+=sy*1.5f;
        }
        if(use_steer && t%100==0 && nmarks<10){marks_x[nmarks]=x;marks_y[nmarks]=y;nmarks++;}
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
    float speeds[NSPEEDS]={1.0f,4.0f,8.0f,16.0f,32.0f};
    printf("=== TERRITORY STEERING vs SPEED: Law 221 ===\nN=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    printf("%-6s | %-10s | %-10s | %-8s\n","Speed","PureScript","Steer","Lift");
    printf("--------|------------|------------|----------\n");
    for(int s=0;s<NSPEEDS;s++){
        float r0=0,r1=0;
        for(int st=0;st<2;st++){
            float ts=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,st,(unsigned int)(42+tr*1111+s*111+st*11));
                cudaDeviceSynchronize();
                float hs[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                float avg=0;for(int i=0;i<N;i++)avg+=hs[i];ts+=avg/N;
            }
            if(st==0)r0=ts/TRIALS;else r1=ts/TRIALS;
        }
        float lift=(r0>0)?(r1-r0)/r0*100:0;
        printf("%-6dx | %8.3f   | %8.3f   | %+6.1f%%\n",(int)speeds[s],r0,r1,lift);
    }
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}