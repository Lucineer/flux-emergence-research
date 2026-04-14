#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
#define NDCS 8
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[NDCS],dcs_y[NDCS],dcs_v[NDCS],dcs_n[1];
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__device__ void add_dcs(int fx,int fy){
    int slot=atomicAdd(dcs_n,1)%NDCS;
    dcs_x[slot]=fx;dcs_y[slot]=fy;dcs_v[slot]=1;
}
__global__ void ss(int step,int nslots){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    int ns=(nslots<1)?0:nslots;
    if(ns>0){
        int bdd=999999,bds=-1;
        for(int s=0;s<ns;s++){if(!dcs_v[s])continue;int d=td(ax[i],ay[i],dcs_x[s],dcs_y[s]);if(d<bd){bdd=d;bds=s;}}
        if(bds>=0&&bdd<g2*4&&bd>bdd){
            int dx=dcs_x[bds]-ax[i],dy=dcs_y[bds]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            bd=999999;bf=-1;
            for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
            if(bdd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;add_dcs(fx[bf],fy[bf]);}
            else if(bf>=0&&bd<=g2&&falive[bf]){falive[bf]=0;acol[i]++;add_dcs(fx[bf],fy[bf]);}
            return;
        }
    }
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;add_dcs(fx[bf],fy[bf]);}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Multi-Point DCS Slots ===\\n");
    printf("4096 agents, 400 food, grab=12\\n");
    printf("Slots | Collection | vs None\\n");
    printf("-----------------------------\\n");
    float pa[5];
    for(int si=0;si<5;si++){
        pa[si]=0;
        for(int trial=0;trial<3;trial++){
            int slots=(si==0)?0:((si==1)?1:((si==2)?2:((si==3)?4:8)));
            int z[NDCS];for(int j=0;j<NDCS;j++)z[j]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int)*NDCS);
            int zn=0;cudaMemcpyToSymbol(dcs_n,&zn,sizeof(int));
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,slots);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[si]+=(float)t/NA;
        }
        pa[si]/=3;
    }
    int slots[]={0,1,2,4,8};
    for(int s=0;s<5;s++){
        printf("  %2d   | %9.1f |",slots[s],pa[s]);
        if(s==0)printf(" baseline\\n");else printf(" %.2fx\\n",pa[s]/pa[0]);
    }
    return 0;
}
