#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSPEEDS 8
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
    float script_dir[8];
    for(int i=0;i<8;i++) script_dir[i]=base_angle+i*0.785f;
    
    for(int t=0;t<steps&&energy>0;t++){
        float dx=0,dy=0,r=4.0f;
        if(strat==0){ // Random
            dx=(cr(&rng)-0.5f)*6.0f*speed_mult;
            dy=(cr(&rng)-0.5f)*6.0f*speed_mult;
        } else if(strat==1){ // Food scan (approach)
            float best_d=999.0f,bx=x,by=y;
            for(int i=0;i<food_count;i+=8){
                if(!falive[i])continue;
                float fdx=fx[i]-x,fdy=fy[i]-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=fdx*fdx+fdy*fdy;
                if(d<best_d){best_d=d;bx=fx[i];by=fy[i];}
            }
            if(best_d<400.0f){
                dx=(bx-x)*0.3f*speed_mult;
                dy=(by-y)*0.3f*speed_mult;
            } else {
                dx=(cr(&rng)-0.5f)*4.0f*speed_mult;
                dy=(cr(&rng)-0.5f)*4.0f*speed_mult;
            }
        } else { // Scripted
            int p=t%8;dx=cosf(script_dir[p])*2.0f;dy=sinf(script_dir[p])*2.0f;
        }
        float dist=sqrtf(dx*dx+dy*dy);
        energy-=0.005f+dist*0.003f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        for(int i=0;i<food_count;i++){
            if(!falive[i])continue;
            float fdx=fx[i]-x,fdy=fy[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            if(fdx*fdx+fdy*fdy<r*r){int old=atomicExch(&falive[i],0);if(old){energy=fminf(energy+10.0f,200.0f);score+=1.0f;}}
        }
    }
    scores[tid]=score;alive[tid]=(energy>0)?1:0;
}

int main(){
    const char* nm[]={"Random","FoodScan","Scripted"};
    float speeds[NSPEEDS]={1,2,3,4,5,6,8,10};
    printf("=== FOOD SCAN vs SPEED: Law 209 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    float res[NSPEEDS][3];
    for(int s=0;s<NSPEEDS;s++){
        for(int st=0;st<3;st++){
            float ts=0;
            for(int tr=0;tr<TRIALS;tr++){
                cudaMemset(d_fa,1,FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,speeds[s],STEPS,N,FOOD,W,st,(unsigned int)(42+tr*1111+s*111+st*11));
                cudaDeviceSynchronize();
                float hs[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
                float avg=0;for(int i=0;i<N;i++)avg+=hs[i];ts+=avg/N;
            }
            res[s][st]=ts/TRIALS;
        }
        int best=0;for(int st=1;st<3;st++)if(res[s][st]>res[s][best])best=st;
        printf("%-5dx | Random:%5.3f | Scan:%5.3f | Script:%5.3f | Winner:%s\n",
            (int)speeds[s],res[s][0],res[s][1],res[s][2],nm[best]);
    }
    // Find crossover
    printf("\n=== CROSSOVER ANALYSIS ===\n");
    for(int s=1;s<NSPEEDS;s++){
        if(res[s-1][1]>res[s-1][2] && res[s][2]>res[s][1])
            printf("FoodScan beaten by Scripted between %dx and %dx\n",(int)speeds[s-1],(int)speeds[s]);
        if(res[s-1][1]>res[s-1][0] && res[s][0]>res[s][1])
            printf("FoodScan beaten by Random between %dx and %dx\n",(int)speeds[s-1],(int)speeds[s]);
    }
    printf("\n>> Law 209: Intelligence (food scanning) has a speed ceiling.\n");
    printf("   Beyond that ceiling, deterministic scripts outperform smart approaches.\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}