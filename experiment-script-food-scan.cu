#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSTRATS 4
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w,
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
        
        if(strat==0){ // Pure scripted
            int p=t%8;dx=cosf(script_dir[p])*2.0f;dy=sinf(script_dir[p])*2.0f;
        } else if(strat==1){ // Scripted + full food scan
            float best_d=1e10f,bx=x,by=y;
            for(int i=0;i<food_count;i++){
                if(!falive[i])continue;
                float fdx=fx[i]-x,fdy=fy[i]-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=fdx*fdx+fdy*fdy;
                if(d<best_d){best_d=d;bx=fx[i];by=fy[i];}
            }
            if(best_d<900.0f){ // food within 30 units — approach
                float fdx=bx-x,fdy=by-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=sqrtf(fdx*fdx+fdy*fdy);
                if(d>0.1f){dx=fdx/d*2.0f;dy=fdy/d*2.0f;}
            } else { // follow script
                int p=t%8;dx=cosf(script_dir[p])*2.0f;dy=sinf(script_dir[p])*2.0f;
            }
        } else if(strat==2){ // Scripted + partial scan (every 4th food)
            float best_d=1e10f,bx=x,by=y;
            for(int i=0;i<food_count;i+=4){
                if(!falive[i])continue;
                float fdx=fx[i]-x,fdy=fy[i]-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=fdx*fdx+fdy*fdy;
                if(d<best_d){best_d=d;bx=fx[i];by=fy[i];}
            }
            if(best_d<900.0f){
                float fdx=bx-x,fdy=by-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=sqrtf(fdx*fdx+fdy*fdy);
                if(d>0.1f){dx=fdx/d*2.0f;dy=fdy/d*2.0f;}
            } else {
                int p=t%8;dx=cosf(script_dir[p])*2.0f;dy=sinf(script_dir[p])*2.0f;
            }
        } else { // Pure food scan (no script)
            float best_d=1e10f,bx=x,by=y;
            for(int i=0;i<food_count;i++){
                if(!falive[i])continue;
                float fdx=fx[i]-x,fdy=fy[i]-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=fdx*fdx+fdy*fdy;
                if(d<best_d){best_d=d;bx=fx[i];by=fy[i];}
            }
            if(best_d<900.0f){
                float fdx=bx-x,fdy=by-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                float d=sqrtf(fdx*fdx+fdy*fdy);
                if(d>0.1f){dx=fdx/d*2.0f;dy=fdy/d*2.0f;}
            } else {
                dx=(cr(&rng)-0.5f)*4.0f;dy=(cr(&rng)-0.5f)*4.0f;
            }
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
    const char* nm[]={"PureScript","Script+Scan","Script+PartScan","PureScan"};
    printf("=== SCRIPT + PERCEPTION HYBRID: Law 212 ===\n");
    printf("N=%d Food=%d Steps=%d Speed=1x Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    
    printf("%-18s | %-8s | %-8s | %-10s\n","Strategy","Score","Surv","FleetTotal");
    printf("-------------------|----------|----------|------------\n");
    float best=0;int best_s=0;
    for(int s=0;s<NSTRATS;s++){
        float ts=0,ta=0,tt=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,STEPS,N,FOOD,W,s,(unsigned int)(42+tr*1111+s*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];
            cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0;float total=0;
            for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];total+=hs[i];}
            ts+=avg/N;ta+=ac;tt+=total;
        }
        ts/=TRIALS;float surv=ta/TRIALS/N*100;tt/=TRIALS;
        if(ts>best){best=ts;best_s=s;}
        printf("%-18s | %6.3f   | %5.1f%% | %8.1f\n",nm[s],ts,surv,tt);
    }
    printf("\n>> Law 212: %s wins at 1x speed\n",nm[best_s]);
    printf("   Key: does adding perception to a script help or hurt?\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}