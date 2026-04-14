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
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int cascade_lens[10000]; // store cascade lengths
__device__ int cascade_count[1];
__device__ int current_uses[1];
__device__ int prev_dcs_x[1],prev_dcs_y[1];

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){
    int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    return dx*dx+dy*dy;
}
__global__ void init_w(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    if(dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            // Count this agent as following DCS
            if(i==0){
                // Thread 0 does bookkeeping each step
                // (not perfect but avoids atomics)
            }
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2&&bf>=0&&falive[bf]){
                falive[bf]=0;acol[i]++;
                dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];
            }
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){
            falive[bf]=0;acol[i]++;
            dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;
        }
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== DCS Point Attraction Tracking ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n\\n");
    int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
    init_w<<<32,BLK>>>(42);init_f<<<fb,BLK>>>(999);cudaDeviceSynchronize();
    int near_count=0,total_checks=0,dcs_active=0;
    int hax[NA],hay[NA],hdx,hdv;
    for(int s=0;s<STEPS;s++){
        ss<<<32,BLK>>>(s);do_resp<<<fb,BLK>>>(0);cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(hax,ax,sizeof(int)*NA);
        cudaMemcpyFromSymbol(hay,ay,sizeof(int)*NA);
        cudaMemcpyFromSymbol(&hdv,dcs_v,sizeof(int));
        cudaMemcpyFromSymbol(&hdx,dcs_x,sizeof(int));
        int hdy;cudaMemcpyFromSymbol(&hdy,dcs_y,sizeof(int));
        if(hdv){dcs_active++;int nc=0;
        for(int i=0;i<NA;i++){
            int dx=hax[i]-hdx;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
            int dy=hay[i]-hdy;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx*dx+dy*dy<=576)nc++;
        }
        near_count+=nc;total_checks++;
        }
    }
    printf("DCS active for %d of %d steps (%.1f%%)\\n",dcs_active,STEPS,100.0*dcs_active/STEPS);
    printf("Avg agents near DCS point: %.1f\\n",(float)near_count/total_checks);
    printf("Expected random: %.1f agents\\n",(float)NA*576.0/(SZ*SZ));
    printf("Attraction ratio: %.1fx random\\n",(float)near_count/total_checks/((float)NA*576.0/(SZ*SZ)));
    return 0;
}
