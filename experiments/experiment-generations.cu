#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],agen[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int gen_length; // steps per generation
__device__ int global_gen; // current generation number
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;agen[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    // Check if new generation
    if(gen_length>0&&step>0&&step%gen_length==0){
        agen[i]++;acol[i]=0;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;
    }
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){if(falive[bf])falive[bf]=0;acol[i]++;}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Generational Turnover ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n");
    printf("GenLen | Collections | Total | Gens | Avg/gen\\n");
    printf("------------------------------------------\\n");
    int gens[]={0,100,300,500,1000,1500,3000};
    for(int gi=0;gi<7;gi++){
        cudaMemcpyToSymbol(gen_length,&gens[gi],sizeof(int));
        float pa=0;
        for(int trial=0;trial<2;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA],ha[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);cudaMemcpyFromSymbol(ha,agen,sizeof(int)*NA);
            long t=0,maxgen=0;for(int i=0;i<NA;i++){t+=hc[i];if(ha[i]>maxgen)maxgen=ha[i];}pa+=(float)t/NA;
        }
        pa/=2;
        int ng=(gens[gi]==0)?1:(STEPS/gens[gi]);
        printf("  %4d  | %10.0f | %ld | %2d | %.0f\\n",gens[gi],pa,(long)(pa*NA),ng,pa/ng);
    }
    return 0;
}
