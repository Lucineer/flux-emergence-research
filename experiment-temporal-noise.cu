#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 256
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define GRID 64
#define NMODES 3
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, int grid_w, int mode, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    int is_emitter = (tid < n/2) ? 1 : 0;
    float noise = is_emitter ? 1.0f : 0.0f;
    int traces[GRID*GRID];
    for(int i=0;i<GRID*GRID;i++)traces[i]=0;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        dx+=(cr(&rng)-0.5f)*2.0f*noise;dy+=(cr(&rng)-0.5f)*2.0f*noise;
        // Observer steering
        if(!is_emitter && mode>0){
            int gx=(int)(x/w*grid_w)%grid_w;int gy=(int)(y/w*grid_w)%grid_w;
            if(mode==1){
                // Real-time: steer toward emitter density (use shared positions via traces)
                atomicAdd(&traces[gy*grid_w+gx],1);
            }
            if(mode==2){
                // Historical traces
                atomicAdd(&traces[gy*grid_w+gx],1);
            }
            float sx=0,sy=0;
            for(int dy2=-2;dy2<=2;dy2++)for(int dx2=-2;dx2<=2;dx2++){
                if(dx2==0&&dy2==0)continue;
                int nx2=(gx+dx2+grid_w)%grid_w,ny2=(gy+dy2+grid_w)%grid_w;
                int v=traces[ny2*grid_w+nx2];
                sx+=dx2*(float)v;sy+=dy2*(float)v;
            }
            if(mode==2){
                // Historical: steer away (avoid where we've been)
                dx-=sx*0.1f;dy-=sy*0.1f;
            } else {
                // Real-time: steer toward emitter density
                dx+=sx*0.1f;dy+=sy*0.1f;
            }
        }
        if(is_emitter){
            int gx=(int)(x/w*grid_w)%grid_w;int gy=(int)(y/w*grid_w)%grid_w;
            atomicAdd(&traces[gy*grid_w+gx],1);
        }
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
    const char* nm[]={"NoTrace","Realtime","Historical"};
    printf("=== TEMPORAL NOISE: Law 261 ===\nN=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    for(int m=0;m<NMODES;m++){
        float ts=0,ta=0,tt=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,GRID,m,(unsigned int)(42+tr*1111+m*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;float total=0;
            for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];total+=hs[i];}
            ts+=avg/N;ta+=ac;tt+=total;
        }
        ts/=TRIALS;float surv=ta/TRIALS/N*100;tt/=TRIALS;
        printf("%-12s: score=%.3f surv=%.1f%% fleet=%.1f\n",nm[m],ts,surv,tt);
    }
    printf("\n>> Law 261: Real-time vs historical noise observation\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}