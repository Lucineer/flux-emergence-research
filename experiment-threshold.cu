#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSPEEDS 13
#define TRIALS 5
#define GRAB 4.0f

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w, int strat, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    for (int i=0;i<8;i++) script_dir[i] = base_angle + i*0.785f;
    float move_spd=2.0f, grab_r=GRAB, last_adapt=-999.0f;
    for (int t=0;t<steps&&energy>0;t++){
        float dx=0,dy=0,r=GRAB;
        if(strat==0){dx=(cr(&rng)-0.5f)*6.0f*speed_mult;dy=(cr(&rng)-0.5f)*6.0f*speed_mult;}
        else if(strat==1){
            if(t-last_adapt>50.0f){move_spd=1.0f+cr(&rng)*4.0f;grab_r=1.0f+cr(&rng)*6.0f;last_adapt=t;}
            float a=cr(&rng)*6.2832f;dx=cosf(a)*move_spd*speed_mult;dy=sinf(a)*move_spd*speed_mult;r=grab_r;
        } else {
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
    float speeds[NSPEEDS]={1,2,3,4,5,6,8,10,12,16,20,24,32};
    const char* nm[]={"Random","Adaptive","Scripted"};
    printf("=== SPEED THRESHOLD SWEEP: Law 203 ===\n");
    printf("N=%d Food=%d Steps=%d Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    float srv[NSPEEDS][3];
    for(int s=0;s<NSPEEDS;s++)for(int st=0;st<3;st++){
        float ta=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,speeds[s],STEPS,N,FOOD,W,st,(unsigned int)(42+tr*1111+s*111+st*11));
            cudaDeviceSynchronize();
            int ha[N];cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            int ac=0;for(int i=0;i<N;i++)ac+=ha[i];ta+=ac;
        }
        srv[s][st]=ta/TRIALS/N*100;
    }
    printf("%-6s | %-8s | %-8s | %-8s | %-10s\n","Speed","Random","Adaptive","Scripted","Gap");
    printf("-------|----------|----------|----------|------------\n");
    float threshold = 0;
    for(int s=0;s<NSPEEDS;s++){
        float gap = srv[s][2]-srv[s][0];
        printf("%-6dx | %6.1f%% | %6.1f%% | %6.1f%% | %+7.1f%%\n",(int)speeds[s],srv[s][0],srv[s][1],srv[s][2],gap);
        if(srv[s][0]<50.0f && srv[s][2]>90.0f && threshold==0) threshold=speeds[s];
    }
    printf("\n=== THRESHOLD ANALYSIS ===\n");
    if(threshold>0) printf(">> Law 203: Speed threshold at ~%dx — scripted survives while reactive collapses\n",threshold);
    // Find exact crossover
    for(int s=1;s<NSPEEDS;s++){
        if(srv[s-1][0]>50.0f && srv[s][0]<50.0f)
            printf(">> Reactive collapse between %dx and %dx\n",(int)speeds[s-1],(int)speeds[s]);
    }
    printf("\nScripted survival at each speed: ");
    for(int s=0;s<NSPEEDS;s++) printf("%dx=%.0f%% ",(int)speeds[s],srv[s][2]);
    printf("\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}