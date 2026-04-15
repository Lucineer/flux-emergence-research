#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define STEPS 3000
#define W 256
#define BLK 128
#define NMODES 2
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate_generalist(float *scores, int *alive,
    float *fx_a, float *fy_a, int *fa_a,
    float *fx_b, float *fy_b, int *fa_b,
    int steps, int n, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        float dist=sqrtf(dx*dx+dy*dy);energy-=0.005f+dist*0.003f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        // Eat type A (energy 5)
        for(int i=0;i<200;i++){
            if(!fa_a[i])continue;
            float fdx=fx_a[i]-x,fdy=fy_a[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            if(fdx*fdx+fdy*fdy<16.0f){int old=atomicExch(&fa_a[i],0);if(old){energy=fminf(energy+5.0f,200.0f);score+=1.0f;}}
        }
        // Eat type B (energy 15)
        for(int i=0;i<200;i++){
            if(!fa_b[i])continue;
            float fdx=fx_b[i]-x,fdy=fy_b[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            if(fdx*fdx+fdy*fdy<16.0f){int old=atomicExch(&fa_b[i],0);if(old){energy=fminf(energy+15.0f,200.0f);score+=1.0f;}}
        }
    }
    scores[tid]=score;alive[tid]=(energy>0)?1:0;
}

__global__ void simulate_specialist(float *scores, int *alive,
    float *fx_a, float *fy_a, int *fa_a,
    float *fx_b, float *fy_b, int *fa_b,
    int steps, int n, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    int type = tid < n/2 ? 0 : 1; // 0=A specialist, 1=B specialist
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        float dist=sqrtf(dx*dx+dy*dy);energy-=0.005f+dist*0.003f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        if(type==0){
            for(int i=0;i<200;i++){
                if(!fa_a[i])continue;
                float fdx=fx_a[i]-x,fdy=fy_a[i]-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                if(fdx*fdx+fdy*fdy<16.0f){int old=atomicExch(&fa_a[i],0);if(old){energy=fminf(energy+5.0f,200.0f);score+=1.0f;}}
            }
        } else {
            for(int i=0;i<200;i++){
                if(!fa_b[i])continue;
                float fdx=fx_b[i]-x,fdy=fy_b[i]-y;
                if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
                if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
                if(fdx*fdx+fdy*fdy<16.0f){int old=atomicExch(&fa_b[i],0);if(old){energy=fminf(energy+15.0f,200.0f);score+=1.0f;}}
            }
        }
    }
    scores[tid]=score;alive[tid]=(energy>0)?1:0;
}

int main(){
    printf("=== MULTI-RESOURCE: Law 252 ===\nN=%d Steps=%d Trials=%d\n",N,STEPS,TRIALS);
    printf("200 food_A (5 energy) + 200 food_B (15 energy)\n\n");
    float *d_s,*d_fx_a,*d_fy_a,*d_fx_b,*d_fy_b;int *d_al,*d_fa_a,*d_fa_b;
    cudaMalloc(&d_s,N*sizeof(float));cudaMalloc(&d_al,N*sizeof(int));
    cudaMalloc(&d_fx_a,200*sizeof(float));cudaMalloc(&d_fy_a,200*sizeof(float));cudaMalloc(&d_fa_a,200*sizeof(int));
    cudaMalloc(&d_fx_b,200*sizeof(float));cudaMalloc(&d_fy_b,200*sizeof(float));cudaMalloc(&d_fa_b,200*sizeof(int));
    float hfx[200],hfy[200];srand(42);
    for(int i=0;i<200;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    int blk=(N+BLK-1)/BLK;
    // Generalist
    float gs=0,ga=0;
    for(int tr=0;tr<TRIALS;tr++){
        cudaMemcpy(d_fx_a,hfx,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy_a,hfy,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fx_b,hfx,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy_b,hfy,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemset(d_fa_a,1,200*sizeof(int));cudaMemset(d_fa_b,1,200*sizeof(int));
        simulate_generalist<<<blk,BLK>>>(d_s,d_al,d_fx_a,d_fy_a,d_fa_a,d_fx_b,d_fy_b,d_fa_b,STEPS,N,W,(unsigned int)(42+tr*1111));
        cudaDeviceSynchronize();
        float hs[N];int ha[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(ha,d_al,N*sizeof(int),cudaMemcpyDeviceToHost);
        float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
        gs+=avg/N;ga+=ac;
    }
    gs/=TRIALS;float gsurv=ga/TRIALS/N*100;
    printf("Generalist (512): avg_food=%.3f survival=%.1f%%\n",gs,gsurv);
    // Specialist
    float ss=0,sa=0;
    for(int tr=0;tr<TRIALS;tr++){
        cudaMemcpy(d_fx_a,hfx,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy_a,hfy,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fx_b,hfx,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy_b,hfy,200*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemset(d_fa_a,1,200*sizeof(int));cudaMemset(d_fa_b,1,200*sizeof(int));
        simulate_specialist<<<blk,BLK>>>(d_s,d_al,d_fx_a,d_fy_a,d_fa_a,d_fx_b,d_fy_b,d_fa_b,STEPS,N,W,(unsigned int)(42+tr*1111));
        cudaDeviceSynchronize();
        float hs[N];int ha[N];cudaMemcpy(hs,d_s,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(ha,d_al,N*sizeof(int),cudaMemcpyDeviceToHost);
        float avg=0;int ac=0;for(int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
        ss+=avg/N;sa+=ac;
    }
    ss/=TRIALS;float ssurv=sa/TRIALS/N*100;
    printf("Specialist (256+256): avg_food=%.3f survival=%.1f%%\n",ss,ssurv);
    printf("\n>> Law 252: Generalist vs specialist with multiple resource types\n");
    cudaFree(d_s);cudaFree(d_al);cudaFree(d_fx_a);cudaFree(d_fy_a);cudaFree(d_fa_a);cudaFree(d_fx_b);cudaFree(d_fy_b);cudaFree(d_fa_b);
    return 0;
}