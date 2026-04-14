#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
#define MAX_GUILDS 64

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[MAX_GUILDS],dcs_y[MAX_GUILDS],dcs_v[MAX_GUILDS];
__device__ float gr=12.0f;
__device__ int N_GUILDS;

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
__global__ void ss(int step,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g=i%N_GUILDS;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(use_dcs&&dcs_v[g]){
        int dd=td(ax[i],ay[i],dcs_x[g],dcs_y[g]);
        if(dd<(int)(gr*gr*4)&&bd>dd){
            int dx=dcs_x[g]-ax[i],dy=dcs_y[g]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=(int)(gr*gr)&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[g]=fx[bf];dcs_y[g]=fy[bf];dcs_v[g]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=(int)(gr*gr)){
        if(falive[bf]){falive[bf]=0;acol[i]++;int g=i%N_GUILDS;dcs_x[g]=fx[bf];dcs_y[g]=fy[bf];dcs_v[g]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Guild Count Sweep: 4096 agents, 400 food ===\n");
    printf("Guilds | Agents/Guild | DCS/agent | NoDCS/agent | Lift\n");
    printf("----------------------------------------------------\n");
    int guilds[]={1,2,4,8,16,32,64};
    int ng=7;
    float nodcs=0;
    for(int trial=0;trial<3;trial++){
        int z[MAX_GUILDS];for(int i=0;i<MAX_GUILDS;i++)z[i]=0;
        cudaMemcpyToSymbol(dcs_v,z,sizeof(int)*MAX_GUILDS);
        init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,0);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();
        int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
        long t=0;for(int i=0;i<NA;i++)t+=hc[i];nodcs+=(float)t/NA;
    }
    nodcs/=3;
    for(int gi=0;gi<ng;gi++){
        int ng_val=guilds[gi];
        cudaMemcpyToSymbol(N_GUILDS,&ng_val,sizeof(int));
        float pa=0;
        for(int trial=0;trial<3;trial++){
            int z[MAX_GUILDS];for(int i=0;i<MAX_GUILDS;i++)z[i]=0;
            cudaMemcpyToSymbol(dcs_v,z,sizeof(int)*MAX_GUILDS);
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,1);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
        }
        pa/=3;
        printf("%5d | %12d | %9.1f | %11.1f | %.2fx\n",ng_val,NA/ng_val,pa,nodcs,pa/nodcs);
    }
    return 0;
}
