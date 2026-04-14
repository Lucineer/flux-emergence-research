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
__device__ int grab_wasted[1]; // times agent arrived at food but already collected
__device__ int grab_success[1]; // times agent successfully collected
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;grab_success[0]++;}
        else{grab_wasted[0]++;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Food Competition Analysis ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n");
    int z[2]={0,0};cudaMemcpyToSymbol(grab_wasted,z,sizeof(int));cudaMemcpyToSymbol(grab_success,z+1,sizeof(int));
    init_w<<<32,BLK>>>(42);init_f<<<fb,BLK>>>(999);cudaDeviceSynchronize();
    for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
    cudaDeviceSynchronize();
    int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
    int gw,gs;cudaMemcpyFromSymbol(&gw,grab_wasted,sizeof(int));cudaMemcpyFromSymbol(&gs,grab_success,sizeof(int));
    long t=0;for(int i=0;i<NA;i++)t+=hc[i];
    printf("Total collections: %ld\\n",t);
    printf("Successful grabs: %d\\n",gs);
    printf("Wasted grabs (arrived, already taken): %d\\n",gw);
    printf("Waste rate: %.1f%%\\n",100.0*gw/(gw+gs));
    printf("Agents per food item: %.1f\\n",(float)NA/FOOD);
    printf("\\nWith %.1f agents per food, %.1f%% of grab attempts waste\\n",(float)NA/FOOD,100.0*gw/(gw+gs));
    return 0;
}
