#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],aspeed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed,int spd){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;aspeed[i]=spd;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int spd=aspeed[i],g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    int moved=0,collected=0;
    if(dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+(dx*spd)/2);ay[i]=to(ay[i]+(dy*spd)/2);moved=1;}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;collected=1;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
        }
    }
    if(!moved){
        if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;collected=1;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
        else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx*spd);ay[i]=to(ay[i]+dy*spd);moved=1;}}
    }
    // Inverse learning: slow down on success (stay near food)
    if(mode==1&&collected&&spd>1)aspeed[i]=spd-1;
    if(mode==1&&!collected&&step%50==0&&spd<4)aspeed[i]=spd+1;
    // Mode 2: converge to speed 1 (always slow down)
    if(mode==2&&spd>1&&step%20==0)aspeed[i]=spd-1;
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Inverse Learning: Slow Down on Success ===\\n");
    printf("4096 agents, 400 food, grab=12\\n");
    float pa[5];
    char*names[]={"Fixed-1","Fixed-2","Fixed-3","Learn-inverse","Always-slow"};
    for(int mi=0;mi<5;mi++){
        pa[mi]=0;
        for(int trial=0;trial<3;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            int ispd=(mi<3)?(mi+1):3;
            int mode=(mi>=3)?(mi-2):0;
            init_w<<<32,BLK>>>(42+trial,ispd);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mi]+=(float)t/NA;
        }
        pa[mi]/=3;
    }
    for(int m=0;m<5;m++)printf("  %-16s | %.0f/agent\\n",names[m],pa[m]);
    return 0;
}
