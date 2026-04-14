#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],aalive[NA];
__device__ float ahp[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ float gr=12.0f;

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){
    int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    return dx*dx+dy*dy;
}
__global__ void init_w(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;
    ahp[i]=1.0f;aalive[i]=1;acol[i]=0;
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step,float pc){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NA||!aalive[i])return;
    float e=ahp[i];
    int bd=999999,bf=-1;
    if(e>pc){for(int f=0;f<FOOD;f++){if(!falive[f])continue;
    int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}e-=pc;}
    if(bf>=0&&bd<=(int)(gr*gr)){
        if(falive[bf]){falive[bf]=0;acol[i]++;e+=0.15f;}
    }else if(bf>=0&&e>pc){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
    e-=0.001f;if(e<=0.0f){aalive[i]=0;}
    ahp[i]=e;
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Perception Cost Sweep ===\n");
    printf("Cost   | Collection/agent | Alive/4096\n");
    printf("---------------------------------------\n");
    float costs[]={0.0f,0.001f,0.002f,0.004f,0.008f,0.01f,0.015f,0.02f,0.03f,0.05f,0.08f,0.1f};
    int nc=12;
    for(int ci=0;ci<nc;ci++){
        float pa=0;int alive=0;
        for(int trial=0;trial<3;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,costs[ci]);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA],ha[NA];
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            cudaMemcpyFromSymbol(ha,aalive,sizeof(int)*NA);
            long t=0;int al=0;for(int i=0;i<NA;i++){t+=hc[i];al+=ha[i];}
            pa+=(float)t/NA;alive+=al;
        }
        pa/=3;alive/=3;
        printf("%.4f  | %15.1f | %4d %s\n",costs[ci],pa,alive,alive<4000?"DYING":"");
    }
    return 0;
}
