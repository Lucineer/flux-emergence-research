#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 2000
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to_t(int v){return((v%SZ)+SZ)%SZ;}
__device__ int clamp(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}
__device__ int td_t(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__device__ int td_b(int x1,int y1,int x2,int y2){int dx=x1-x2,dy=y1-y2;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss_t(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td_t(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){if(falive[bf])falive[bf]=0;acol[i]++;}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to_t(ax[i]+dx);ay[i]=to_t(ay[i]+dy);}}
}
__global__ void ss_b(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td_b(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){if(falive[bf])falive[bf]=0;acol[i]++;}
    else if(bf>=0){ax[i]=clamp(ax[i]+(fx[bf]-ax[i]),0,SZ-1);ay[i]=clamp(ay[i]+(fy[bf]-ay[i]),0,SZ-1);}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Toroidal vs Bounded World ===\\n");
    printf("4096 agents, 400 food, grab=12, 2000 steps\\n");
    float pt=0,pb=0;
    for(int trial=0;trial<3;trial++){
        init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){ss_t<<<32,BLK>>>(s);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
        long t=0;for(int i=0;i<NA;i++)t+=hc[i];pt+=(float)t/NA;
        init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){ss_b<<<32,BLK>>>(s);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
        t=0;for(int i=0;i<NA;i++)t+=hc[i];pb+=(float)t/NA;
    }
    pt/=3;pb/=3;
    printf("  Toroidal: %.0f/agent\\n",pt);
    printf("  Bounded:  %.0f/agent (%.2fx)\\n",pb,pb/pt);
    return 0;
}
