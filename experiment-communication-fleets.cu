#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NF 128
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(float *scores_a, float *scores_b, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, int b_reads_marks, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    // Fleet A: Steering (marks)
    {
        unsigned int rng = seed + tid * 997;
        float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
        float base_angle = tid * 2.39996f;
        float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
        float marks_x[10], marks_y[10];int nmarks=0;
        for(int t=0;t<steps&&energy>0;t++){
            int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
            float sx=0,sy=0;
            for(int m=0;m<nmarks;m++){
                float mx=marks_x[m]-x,my=marks_y[m]-y;
                if(mx>w/2)mx-=w;if(mx<-w/2)mx+=w;
                if(my>w/2)my-=w;if(my<-w/2)my+=w;
                float d=mx*mx+my*my;
                if(d<100.0f&&d>0.01f){sx-=mx/sqrtf(d)*0.5f;sy-=my/sqrtf(d)*0.5f;}
            }
            dx+=sx;dy+=sy;
            if(t%100==0&&nmarks<10){marks_x[nmarks]=x;marks_y[nmarks]=y;nmarks++;}
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
        scores_a[tid]=score;
    }
    // Fleet B: PureScript or reads marks
    {
        unsigned int rng = seed + (tid+n) * 997;
        float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
        float base_angle = (tid+n) * 2.39996f;
        float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
        float marks_x[10], marks_y[10];int nmarks=0;
        for(int t=0;t<steps&&energy>0;t++){
            int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
            if(b_reads_marks){
                float sx=0,sy=0;
                for(int m=0;m<nmarks;m++){
                    float mx=marks_x[m]-x,my=marks_y[m]-y;
                    if(mx>w/2)mx-=w;if(mx<-w/2)mx+=w;
                    if(my>w/2)my-=w;if(my<-w/2)my+=w;
                    float d=mx*mx+my*my;
                    if(d<100.0f&&d>0.01f){sx-=mx/sqrtf(d)*0.5f;sy-=my/sqrtf(d)*0.5f;}
                }
                dx+=sx;dy+=sy;
                if(t%100==0&&nmarks<10){marks_x[nmarks]=x;marks_y[nmarks]=y;nmarks++;}
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
        scores_b[tid]=score;
    }
}

int main(){
    printf("=== CROSS-FLEET COMMUNICATION: Law 253 ===\nFleet=%d Food=%d Steps=%d Trials=%d\n\n",NF,FOOD,STEPS,TRIALS);
    float *d_sa,*d_sb,*d_fx,*d_fy;int *d_fa;
    cudaMalloc(&d_sa,NF*sizeof(float));cudaMalloc(&d_sb,NF*sizeof(float));
    cudaMalloc(&d_fx,FOOD*sizeof(float));cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(NF+BLK-1)/BLK;
    for(int mode=0;mode<2;mode++){
        float ta=0,tb=0,tt=0;
        for(int tr=0;tr<TRIALS;tr++){
            cudaMemset(d_fa,1,FOOD*sizeof(int));
            simulate<<<blk,BLK>>>(d_sa,d_sb,d_fx,d_fy,d_fa,STEPS,NF,FOOD,W,mode,(unsigned int)(42+tr*1111+mode*111));
            cudaDeviceSynchronize();
            float hsa[NF],hsb[NF];cudaMemcpy(hsa,d_sa,NF*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hsb,d_sb,NF*sizeof(float),cudaMemcpyDeviceToHost);
            float avga=0,avgb=0;for(int i=0;i<NF;i++){avga+=hsa[i];avgb+=hsb[i];}
            ta+=avga/NF;tb+=avgb/NF;tt+=(avga+avgb)/NF;
        }
        ta/=TRIALS;tb/=TRIALS;tt/=TRIALS;
        printf("Mode %d: A(steer)=%.3f B(%s)=%.3f total=%.3f\n",
            mode,ta,mode?"reads":"pure",tb,tt);
    }
    printf("\n>> Law 253: Cross-fleet communication effect\n");
    cudaFree(d_sa);cudaFree(d_sb);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);
    return 0;
}