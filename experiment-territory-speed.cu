#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSPEEDS 4
#define NMODES 3
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w,
    int mode, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    // Territory: agents mark their position every 50 steps
    float marks[20]; // up to 20 territory marks
    int nmarks = 0;
    
    for(int t=0;t<steps&&energy>0;t++){
        float dx=(cr(&rng)-0.5f)*6.0f*speed_mult;
        float dy=(cr(&rng)-0.5f)*6.0f*speed_mult;
        
        if(mode==1 || mode==2){
            // Check territory: steer away from (mode 1) or toward (mode 2) own marks
            float steer_x=0,steer_y=0;
            for(int m=0;m<nmarks;m++){
                float mx=marks[m]-x,my=marks[(m+1)%20+20]-y;
                if(mx>w/2)mx-=w;if(mx<-w/2)mx+=w;
                if(my>w/2)my-=w;if(my<-w/2)my+=w;
                float md=mx*mx+my*my;
                if(md<400.0f && md>0.01f){ // within 20 units
                    float sign=(mode==1)?-1.0f:1.0f;
                    steer_x+=sign*mx/sqrtf(md);
                    steer_y+=sign*my/sqrtf(md);
                }
            }
            dx+=steer_x*0.5f;
            dy+=steer_y*0.5f;
        }
        
        float dist=sqrtf(dx*dx+dy*dy);
        energy-=0.005f+dist*0.003f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        
        // Mark territory periodically
        if(t%50==0 && nmarks<20){
            marks[nmarks]=x;
            marks[(nmarks+1)%20+20]=y;
            nmarks++;
        }
        
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
    const char* modes[]={"None","Avoid","Prefer"};
    float speeds[NSPEEDS]={1.0f,4.0f,8.0f,16.0f};
    printf("=== TERRITORY AT HIGH SPEED: Law 211 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    for(int s=0;s<NSPEEDS;s++){
        printf("--- Speed %dx ---\n",(int)speeds[s]);
        float best_score=0;int best_mode=0;
        for(int m=0;m<NMODES;m++){
            float ts=0,ta=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,speeds[s],STEPS,N,FOOD,W,m,(unsigned int)(42+tr*1111+s*111+m*11));
                cudaDeviceSynchronize();
                float hs[N];int ha[N];
                cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
                float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
                ts+=avg/N;ta+=ac;
            }
            ts/=TRIALS;float surv=ta/TRIALS/N*100;
            if(ts>best_score){best_score=ts;best_mode=m;}
            printf("  %s: score=%.3f survival=%.1f%%\n",modes[m],ts,surv);
        }
        printf("  Winner: %s\n\n",modes[best_mode]);
    }
    printf(">> Law 211: Territory avoidance effectiveness vs speed\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}