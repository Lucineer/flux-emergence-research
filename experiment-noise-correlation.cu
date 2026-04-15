#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NF 128
#define FOOD 400
#define W 256
#define BLK 128
#define GRID 64
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void phase1_traces(int *traces, int steps, int n, int w, int grid_w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        dx+=(cr(&rng)-0.5f)*2.0f;dy+=(cr(&rng)-0.5f)*2.0f;
        x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);
        int gx=(int)(x/w*grid_w)%grid_w;int gy=(int)(y/w*grid_w)%grid_w;
        atomicAdd(&traces[gy*grid_w+gx],1);
    }
}

__global__ void build_food_grid(int *food_grid, float *fx, float *fy, int food_count, int w, int grid_w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= food_count) return;
    int gx=(int)(fx[i]/w*grid_w)%grid_w;
    int gy=(int)(fy[i]/w*grid_w)%grid_w;
    atomicAdd(&food_grid[gy*grid_w+gx],1);
}

__global__ void phase2_oracle(float *scores, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + (tid+n) * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = (tid+n) * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        // Oracle: steer toward nearest food
        float best_d2=999999.0f,best_dx=0,best_dy=0;
        for(int i=0;i<food_count;i++){
            if(!falive[i])continue;
            float fdx=fx[i]-x,fdy=fy[i]-y;
            if(fdx>w/2)fdx-=w;if(fdx<-w/2)fdx+=w;
            if(fdy>w/2)fdy-=w;if(fdy<-w/2)fdy+=w;
            float d2=fdx*fdx+fdy*fdy;
            if(d2<best_d2){best_d2=d2;best_dx=fdx;best_dy=fdy;}
        }
        dx+=best_dx*0.1f;dy+=best_dy*0.1f;
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
    scores[tid]=score;
}

__global__ void phase2_traces(float *scores, int *traces, float *fx, float *fy, int *falive,
    int steps, int n, int food_count, int w, int grid_w, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + (tid+n) * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w, energy = 150.0f, score = 0.0f;
    float base_angle = (tid+n) * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    for(int t=0;t<steps&&energy>0;t++){
        int p=t%8;float dx=cosf(dir[p])*2.0f,dy=sinf(dir[p])*2.0f;
        int gx=(int)(x/w*grid_w)%grid_w;int gy=(int)(y/w*grid_w)%grid_w;
        float sx=0,sy=0;
        for(int dy2=-2;dy2<=2;dy2++)for(int dx2=-2;dx2<=2;dx2++){
            if(dx2==0&&dy2==0)continue;
            int nx=(gx+dx2+grid_w)%grid_w,ny=(gy+dy2+grid_w)%grid_w;
            int v=traces[ny*grid_w+nx];
            sx+=dx2*(float)v;sy+=dy2*(float)v;
        }
        dx+=sx*0.1f;dy+=sy*0.1f;
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
    scores[tid]=score;
}

int main(){
    printf("=== NOISE-FOOD CORRELATION: Law 265 ===\nN=%d Food=%d Trials=%d\n\n",NF,FOOD,TRIALS);
    int *d_tr,*d_fg;cudaMalloc(&d_tr,GRID*GRID*sizeof(int));cudaMalloc(&d_fg,GRID*GRID*sizeof(int));
    float *d_s,*d_fx,*d_fy;int *d_fa;
    cudaMalloc(&d_s,NF*sizeof(float));cudaMalloc(&d_fx,FOOD*sizeof(float));
    cudaMalloc(&d_fy,FOOD*sizeof(float));cudaMalloc(&d_fa,FOOD*sizeof(int));
    float hfx[FOOD],hfy[FOOD];srand(42);
    for(int i=0;i<FOOD;i++){hfx[i]=((float)rand()/RAND_MAX)*W;hfy[i]=((float)rand()/RAND_MAX)*W;}
    cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
    int blk=(NF+BLK-1)/BLK;
    int fblk=(FOOD+BLK-1)/BLK;
    // Correlation
    cudaMemset(d_tr,0,GRID*GRID*sizeof(int));cudaMemset(d_fg,0,GRID*GRID*sizeof(int));
    build_food_grid<<<fblk,BLK>>>(d_fg,d_fx,d_fy,FOOD,W,GRID);
    phase1_traces<<<blk,BLK>>>(d_tr,1500,NF,W,GRID,42);
    cudaDeviceSynchronize();
    int htr[GRID*GRID],hfg[GRID*GRID];
    cudaMemcpy(htr,d_tr,GRID*GRID*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(hfg,d_fg,GRID*GRID*sizeof(int),cudaMemcpyDeviceToHost);
    // Pearson correlation
    double sum_tr=0,sum_fg=0,sum_tr2=0,sum_fg2=0,sum_tf=0;
    int cells=GRID*GRID;
    for(int i=0;i<cells;i++){
        sum_tr+=htr[i];sum_fg+=hfg[i];
        sum_tr2+=(double)htr[i]*htr[i];sum_fg2+=(double)hfg[i]*hfg[i];
        sum_tf+=(double)htr[i]*hfg[i];
    }
    double mean_tr=sum_tr/cells,mean_fg=sum_fg/cells;
    double num=sum_tf-cells*mean_tr*mean_fg;
    double den=sqrt((sum_tr2-cells*mean_tr*mean_tr)*(sum_fg2-cells*mean_fg*mean_fg));
    double corr=den>0?num/den:0;
    printf("Trace-Food Pearson correlation: %.4f\n\n",corr);
    // Oracle vs traces
    float oracle_s=0,traces_s=0,ignore_s=0;
    for(int tr=0;tr<TRIALS;tr++){
        // Oracle
        cudaMemset(d_fa,1,FOOD*sizeof(int));
        cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
        phase2_oracle<<<blk,BLK>>>(d_s,d_fx,d_fy,d_fa,1500,NF,FOOD,W,(unsigned int)(42+tr*1111));
        cudaDeviceSynchronize();
        float hs[NF];cudaMemcpy(hs,d_s,NF*sizeof(float),cudaMemcpyDeviceToHost);
        float avg=0;for(int i=0;i<NF;i++)avg+=hs[i];oracle_s+=avg/NF;
        // Traces
        cudaMemset(d_fa,1,FOOD*sizeof(int));
        cudaMemset(d_tr,0,GRID*GRID*sizeof(int));
        cudaMemcpy(d_fx,hfx,FOOD*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_fy,hfy,FOOD*sizeof(float),cudaMemcpyHostToDevice);
        phase1_traces<<<blk,BLK>>>(d_tr,1500,NF,W,GRID,(unsigned int)(42+tr*1111));
        cudaDeviceSynchronize();
        phase2_traces<<<blk,BLK>>>(d_s,d_tr,d_fx,d_fy,d_fa,1500,NF,FOOD,W,GRID,(unsigned int)(42+tr*1111));
        cudaDeviceSynchronize();
        cudaMemcpy(hs,d_s,NF*sizeof(float),cudaMemcpyDeviceToHost);
        avg=0;for(int i=0;i<NF;i++)avg+=hs[i];traces_s+=avg/NF;
    }
    oracle_s/=TRIALS;traces_s/=TRIALS;
    printf("Oracle (perfect food info): %.3f\n",oracle_s);
    printf("Traces (noise-derived):     %.3f\n",traces_s);
    printf("Ratio traces/oracle:        %.2f\n",traces_s/(oracle_s+0.001));
    printf("\n>> Law 265: Do noise traces correlate with food? How good vs oracle?\n");
    cudaFree(d_tr);cudaFree(d_fg);cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);
    return 0;
}