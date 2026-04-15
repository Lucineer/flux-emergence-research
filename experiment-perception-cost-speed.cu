#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NCOSTS 7
#define NSPEEDS 4
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, float perc_cost, int steps, int n, int food_count, int w,
    unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    
    for(int t=0;t<steps&&energy>0;t++){
        float dx,dy;
        // Perception: scan all food, find nearest
        float best_d=1e10f,bx=x,by=y;
        for(int i=0;i<food_count;i++){
            if(!falive[i])continue;
            float fdx=fx[i]-x,fdy=fy[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            float d=fdx*fdx+fdy*fdy;
            if(d<best_d){best_d=d;bx=fx[i];by=fy[i];}
        }
        energy -= perc_cost * food_count; // perception cost scales with food count
        
        if(best_d<900.0f){
            float fdx=bx-x,fdy=by-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            float d=sqrtf(fdx*fdx+fdy*fdy);
            if(d>0.1f){dx=fdx/d*2.0f;dy=fdy/d*2.0f;}
            else{dx=0;dy=0;}
        } else {
            dx=(cr(&rng)-0.5f)*4.0f;dy=(cr(&rng)-0.5f)*4.0f;
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
    float costs[NCOSTS]={0.0f,0.005f,0.01f,0.02f,0.03f,0.05f,0.1f};
    float speeds[NSPEEDS]={1.0f,4.0f,8.0f,16.0f};
    printf("=== PERCEPTION COST vs SPEED: Law 216 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n",N,FOOD,STEPS,TRIALS);
    printf("Perception cost = cost * food_count per step\n\n");
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    for(int sp=0;sp<NSPEEDS;sp++){
        printf("--- Speed %dx ---\n",(int)speeds[sp]);
        printf("Cost    | Score   | Survival\n");
        printf("--------|---------|----------\n");
        for(int c=0;c<NCOSTS;c++){
            float ts=0,ta=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,speeds[sp],costs[c],STEPS,N,FOOD,W,(unsigned int)(42+tr*1111+sp*111+c*11));
                cudaDeviceSynchronize();
                float hs[N];int ha[N];
                cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
                float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
                ts+=avg/N;ta+=ac;
            }
            ts/=TRIALS;float surv=ta/TRIALS/N*100;
            printf("%-7.3f | %6.3f  | %5.1f%%\n",costs[c],ts,surv);
        }
        printf("\n");
    }
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}