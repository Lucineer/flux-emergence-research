#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define SZ 256
#define BLK 128
#define STEPS 2000
#define MAXF 1600
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int *pfx,*pfy,*pfa;
__device__ float *pft;
__device__ int grab_wasted[1],grab_success[1];
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed,int food){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;int s=seed+i*777;pfx[i]=s%SZ;pfy[i]=(s*31)%SZ;pfa[i]=1;pft[i]=0;}
__global__ void do_resp(int unused,int food){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;if(!pfa[i]){pft[i]+=1.0f;if(pft[i]>50.0f){pfa[i]=1;pft[i]=0;}}}
__global__ void ss(int step,int food){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<food;f++){if(!pfa[f])continue;int d=td(ax[i],ay[i],pfx[f],pfy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){
        if(pfa[bf]){pfa[bf]=0;acol[i]++;grab_success[0]++;}
        else grab_wasted[0]++;
    }else if(bf>=0){int dx=pfx[bf]-ax[i],dy=pfy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int maxf=MAXF;
    int *d_fx,*d_fy,*d_fa;float *d_ft;
    cudaMalloc(&d_fx,maxf*sizeof(int));cudaMalloc(&d_fy,maxf*sizeof(int));
    cudaMalloc(&d_fa,maxf*sizeof(int));cudaMalloc(&d_ft,maxf*sizeof(float));
    cudaMemcpyToSymbol(pfx,&d_fx,sizeof(int*));
    cudaMemcpyToSymbol(pfy,&d_fy,sizeof(int*));
    cudaMemcpyToSymbol(pfa,&d_fa,sizeof(int*));
    cudaMemcpyToSymbol(pft,&d_ft,sizeof(float*));
    printf("=== Waste Rate vs Food Density ===\\n");
    printf("4096 agents, 2000 steps\\n");
    printf("Food | Agents/Food | Waste%% | Coll/agent\\n");
    printf("-------------------------------------\\n");
    int foods[]={25,50,100,200,400,800,1600};
    for(int fi=0;fi<7;fi++){
        int food=foods[fi];int fb=(food+BLK-1)/BLK;
        int z[2]={0,0};cudaMemcpyToSymbol(grab_wasted,z,sizeof(int));cudaMemcpyToSymbol(grab_success,z+1,sizeof(int));
        init_w<<<32,BLK>>>(42);init_f<<<fb,BLK>>>(999,food);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,food);do_resp<<<fb,BLK>>>(0,food);if(s%500==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();
        int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
        int gw,gs;cudaMemcpyFromSymbol(&gw,grab_wasted,sizeof(int));cudaMemcpyFromSymbol(&gs,grab_success,sizeof(int));
        long t=0;for(int i=0;i<NA;i++)t+=hc[i];
        printf(" %4d | %10.1f | %5.1f%% | %.0f\\n",food,(float)NA/food,100.0*gw/(gw+gs),(float)t/NA);
    }
    cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);cudaFree(d_ft);
    return 0;
}
