#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
#define MAXK 16

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[MAXK],dcs_y[MAXK],dcs_cnt;

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
__global__ void ss(int step,int topk){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(dcs_cnt>0&&topk>0){
        int kmax=topk;if(kmax>dcs_cnt)kmax=dcs_cnt;if(kmax>MAXK)kmax=MAXK;
        int dcs_bd=999999,dcs_bf=-1;
        for(int k=0;k<kmax;k++){
            int dd=td(ax[i],ay[i],dcs_x[k],dcs_y[k]);
            if(dd<dcs_bd){dcs_bd=dd;dcs_bf=k;}
        }
        if(dcs_bd<g2*4&&bd>dcs_bd){
            int dx=dcs_x[dcs_bf]-ax[i],dy=dcs_y[dcs_bf]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dcs_bd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;}
            int pos=atomicAdd(&dcs_cnt,1)%MAXK;
            dcs_x[pos]=(bf>=0&&falive[bf])?fx[bf]:dcs_x[dcs_bf];
            dcs_y[pos]=(bf>=0&&falive[bf])?fy[bf]:dcs_y[dcs_bf];
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;
        int pos=atomicAdd(&dcs_cnt,1)%MAXK;dcs_x[pos]=fx[bf];dcs_y[pos]=fy[bf];}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== TOP-K DCS Sweep ===\\n");
    printf("4096 agents, 400 food, 256x256, grab=12\\n");
    printf("TOP-K | Collection/agent | vs K=0\\n");
    printf("---------------------------------\\n");
    float base_pa=0;int ks[]={0,1,2,4,8,16};
    for(int ki=0;ki<6;ki++){
        float pa=0;
        for(int trial=0;trial<3;trial++){
            int z=0;cudaMemcpyToSymbol(dcs_cnt,&z,sizeof(int));
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,ks[ki]);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
        }
        pa/=3;
        float base_pa=0;printf("  %2d  | %15.1f |",ks[ki],pa);
        if(ki>0)printf(" %.2fx\\n",pa/base_pa);else{printf(" (baseline)\\n");base_pa=pa;}
    }
    return 0;
}
