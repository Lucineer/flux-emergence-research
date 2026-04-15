#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSTRATS 5
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w,
    int strat, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    
    for(int t=0;t<steps&&energy>0;t++){
        float dx,dy;
        float ang = base_angle;
        
        if(strat==0){ // Fixed script length 8
            ang += (t%8) * 0.785f;
            dx=cosf(ang)*2.0f; dy=sinf(ang)*2.0f;
        } else if(strat==1){ // Fixed script length 32
            ang += (t%32) * 0.19635f;
            dx=cosf(ang)*2.0f; dy=sinf(ang)*2.0f;
        } else if(strat==2){ // Adaptive: energy-based length
            int slen = (energy > 100.0f) ? 8 : (energy > 50.0f) ? 32 : 64;
            ang += (t%slen) * 6.2832f / slen;
            float spd = (energy > 100.0f) ? 3.0f : 2.0f;
            dx=cosf(ang)*spd; dy=sinf(ang)*spd;
        } else if(strat==3){ // Adaptive: speed-based length
            int slen = (energy > 100.0f) ? 4 : 16;
            ang += (t%slen) * 6.2832f / slen;
            dx=cosf(ang)*2.0f; dy=sinf(ang)*2.0f;
        } else { // Adaptive: energy-based speed
            float spd = (energy > 100.0f) ? 4.0f : (energy > 50.0f) ? 2.0f : 1.0f;
            ang += (t%8) * 0.785f;
            dx=cosf(ang)*spd; dy=sinf(ang)*spd;
        }
        
        float dist=sqrtf(dx*dx+dy*dy);
        energy-=0.005f+dist*0.003f;
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
    const char* nm[]={"Script8","Script32","AdaptLen","AdaptShort","AdaptSpeed"};
    float speeds[3]={1.0f,8.0f,32.0f};
    printf("=== ADAPTIVE SCRIPTS: Law 215 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    for(int sp=0;sp<3;sp++){
        printf("--- Speed %dx ---\n",(int)speeds[sp]);
        float best=0;int best_s=0;
        for(int s=0;s<NSTRATS;s++){
            float ts=0,ta=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,speeds[sp],STEPS,N,FOOD,W,s,(unsigned int)(42+tr*1111+sp*111+s*11));
                cudaDeviceSynchronize();
                float hs[N];int ha[N];
                cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
                float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
                ts+=avg/N;ta+=ac;
            }
            ts/=TRIALS;float surv=ta/TRIALS/N*100;
            if(ts>best){best=ts;best_s=s;}
            printf("  %-12s: score=%.3f surv=%.1f%%\n",nm[s],ts,surv);
        }
        printf("  Winner: %s\n\n",nm[best_s]);
    }
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}