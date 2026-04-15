#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NFRACTIONS 7
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores, int *alive, float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w,
    int scripted_count, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    int is_scripted = (tid < scripted_count) ? 1 : 0;
    float base_angle = tid * 2.39996f;
    float script_dir[8];
    for (int i=0;i<8;i++) script_dir[i] = base_angle + i*0.785f;
    float move_spd=2.0f;
    for (int t=0;t<steps&&energy>0;t++){
        float dx=0,dy=0;
        if(is_scripted){
            int p=t%8;dx=cosf(script_dir[p])*2.0f;dy=sinf(script_dir[p])*2.0f;
        } else {
            dx=(cr(&rng)-0.5f)*6.0f*speed_mult;
            dy=(cr(&rng)-0.5f)*6.0f*speed_mult;
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
    float fractions[NFRACTIONS]={0.0f,0.1f,0.25f,0.5f,0.75f,0.9f,1.0f};
    printf("=== MIXED FLEET AT HIGH SPEED: Law 206 ===\n");
    printf("N=%d Food=%d Steps=%d Speed=16x Trials=%d\n\n",N,FOOD,STEPS,TRIALS);
    float *d_s,*d_fx,*d_fy;int *d_a,*d_fa;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_a,N*sizeof(int));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(N+BLK-1)/BLK;
    float fleet_score[NFRACTIONS], fleet_surv[NFRACTIONS];
    float script_score[NFRACTIONS], react_score[NFRACTIONS];
    float script_surv[NFRACTIONS], react_surv[NFRACTIONS];
    for(int f=0;f<NFRACTIONS;f++){
        int sc=(int)(N*fractions[f]);
        float ts=0,ta=0,tss=0,tsa=0,trs=0,tra=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_s,d_a,d_fx,d_fy,d_fa,16.0f,STEPS,N,FOOD,W,sc,(unsigned int)(42+tr*1111+f*111));
            cudaDeviceSynchronize();
            float hs[N];int ha[N];
            cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            float avg=0;int ac=0,ss=0,sa=0,rs=0,ra=0;
            for(int i=0;i<N;i++){
                avg+=hs[i];ac+=ha[i];
                if(i<sc){ss+=hs[i];sa+=ha[i];}
                else if(N-sc>0){rs+=hs[i];ra+=ha[i];}
            }
            ts+=avg/N;ta+=ac;tss+=ss/max(sc,1);tsa+=sa;
            if(N-sc>0){trs+=rs/(N-sc);tra+=ra;}
        }
        fleet_score[f]=ts/TRIALS;fleet_surv[f]=ta/TRIALS/N*100;
        script_score[f]=tss/TRIALS;script_surv[f]=tsa/TRIALS/max(sc,1)*100;
        react_score[f]=trs/TRIALS;react_surv[f]=tra/TRIALS/max(N-(int)(N*fractions[f]),1)*100;
        printf("%4.0f%% scripted | Fleet: %.3f (%.0f%%) | Script: %.3f (%.0f%%) | React: %.3f (%.0f%%)\n",
            fractions[f]*100,fleet_score[f],fleet_surv[f],script_score[f],script_surv[f],react_score[f],react_surv[f]);
    }
    printf("\n=== ANALYSIS ===\n");
    float best=0;int best_f=0;
    for(int f=0;f<NFRACTIONS;f++) if(fleet_score[f]>best){best=fleet_score[f];best_f=f;}
    printf("Optimal scripted fraction: %.0f%% (fleet score %.3f)\n",fractions[best_f]*100,best);
    if(fractions[best_f]>0.5f) printf(">> Law 206: At high speed, majority-scripted fleets dominate\n");
    if(fractions[best_f]<0.5f && fractions[best_f]>0.0f)
        printf(">> Law 206: Small reactive minority + scripted majority = optimal at high speed\n");
    if(fractions[best_f]==1.0f) printf(">> Law 206: Pure scripted fleet is optimal at high speed — no benefit from reactive diversity\n");
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}