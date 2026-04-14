#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int food_mode; // 0=steady, 1=pulsed, 2=wave
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(food_mode==0){if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
    else if(food_mode==1){
        // Pulsed: all food alive at step 0, 300, 600, etc
        if(step%300==0){falive[i]=1;ftimer[i]=0;}
        // No respawn between pulses
    }else{
        // Wave: sine-wave probability of being alive
        float wave=0.5f+0.5f*__sinf(3.14159f*2.0f*step/300.0f);
        if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>20.0f&&rn(&(int){step*777+i})%100<(int)(wave*100)){falive[i]=1;ftimer[i]=0;}}
    }
}
__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){if(falive[bf])falive[bf]=0;acol[i]++;}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Food Availability Patterns ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n");
    float pa[3];
    char*names[]={"Steady-respawn","Pulsed(300)","Sine-wave(300)"};
    for(int mode=0;mode<3;mode++){
        cudaMemcpyToSymbol(food_mode,&mode,sizeof(int));
        pa[mode]=0;
        for(int trial=0;trial<2;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s);do_resp<<<fb,BLK>>>(s);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
        }
        pa[mode]/=2;
    }
    for(int m=0;m<3;m++)printf("  %-20s | %.0f/agent (%.2fx vs steady)\\n",names[m],pa[m],pa[m]/pa[0]);
    return 0;
}
