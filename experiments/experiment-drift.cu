#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 256
#define FOOD 64
#define SZ 64
#define BLK 128
#define STEPS 1500

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ float drift_pct;

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}

__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}

__global__ void apply_drift(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD||!falive[i])return;
    if(rn(&fx[i])<(int)(drift_pct*10000.0f)){
        fx[i]=rn(&fy[i])%SZ;fy[i]=rn(&fx[i])%SZ;
    }
}

__global__ void do_resp(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>30.0f){falive[i]=1;ftimer[i]=0;}}
}

__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
            if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            bd=999999;bf=-1;
            for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            else if(bf>=0&&bd<=g2&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
        if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}

int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Environmental Drift Experiment ===\n");
    printf("256 agents, 64 food, 64x64, grab=12, 1500 steps, 64 trials\n");
    printf("Seed predicts: flat until 3.7%% then collapse\n\n");
    printf("Drift%%  | DCS_Coll | NoDCS_Coll | Ratio | vsBase\n");
    printf("--------+----------+------------+-------+-------\n");

    float drifts[]={0.0f,0.5f,1.0f,1.5f,2.0f,2.5f,3.0f,3.5f,3.7f,4.0f,5.0f,6.0f,7.0f,8.0f,10.0f};
    float base_dcs=0;

    for(int di=0;di<15;di++){
        float dp=drifts[di]/100.0f;
        cudaMemcpyToSymbol(drift_pct,&dp,sizeof(float));
        float dc=0,nc=0;
        for(int t=0;t<64;t++){
            int z=0;cudaMemcpyToSymbol(dcs_v,&z,sizeof(int));
            init_w<<<2,BLK>>>(t*31);init_f<<<fb,BLK>>>(t*47);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<2,BLK>>>(s);apply_drift<<<fb,BLK>>>();do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long total=0;for(int i=0;i<NA;i++)total+=hc[i];dc+=(float)total/NA;

            z=0;cudaMemcpyToSymbol(dcs_v,&z,sizeof(int));
            init_w<<<2,BLK>>>(t*31+1);init_f<<<fb,BLK>>>(t*47+1);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<2,BLK>>>(s);apply_drift<<<fb,BLK>>>();do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            total=0;for(int i=0;i<NA;i++)total+=hc[i];nc+=(float)total/NA;
        }
        dc/=64;nc/=64;
        if(di==0)base_dcs=dc;
        printf("%5.1f%%  | %8.1f | %10.1f | %5.3f | %+6.1f%%\n",
            drifts[di], dc, nc, nc>0?dc/nc:0, base_dcs>0?(dc-base_dcs)/base_dcs*100:0);
    }
    return 0;
}
