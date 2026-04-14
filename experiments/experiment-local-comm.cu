#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 2000
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int tip_x[NA],tip_y[NA],tip_v[NA]; // per-agent tips
__device__ int comm_r2;
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;tip_v[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
// Broadcast tip to neighbors
__global__ void broadcast(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    if(!tip_v[i])return;
    for(int j=0;j<NA;j++){
        if(i==j)continue;
        if(td(ax[i],ay[i],ax[j],ay[j])<=comm_r2){
            tip_x[j]=tip_x[i];tip_y[j]=tip_y[i];tip_v[j]=1;
        }
    }
}
__global__ void ss(int step,int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(mode>=1&&tip_v[i]){
        int dd=td(ax[i],ay[i],tip_x[i],tip_y[i]);
        if(dd<g2*4&&bd>dd){
            int dx=tip_x[i]-ax[i],dy=tip_y[i]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            bd=999999;bf=-1;
            for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;tip_x[i]=fx[bf];tip_y[i]=fy[bf];tip_v[i]=1;}
            else if(bf>=0&&bd<=g2&&falive[bf]){falive[bf]=0;acol[i]++;tip_x[i]=fx[bf];tip_y[i]=fy[bf];tip_v[i]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;tip_x[i]=fx[bf];tip_y[i]=fy[bf];tip_v[i]=1;}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Local vs Global Communication ===\\n");
    printf("4096 agents, 400 food, grab=12, 2000 steps\\n");
    printf("CommRange | Collection | vs None\\n");
    printf("--------------------------------\\n");
    int ranges[]={0,4,8,16,32,64,128,256};
    float pa[8];
    for(int ri=0;ri<8;ri++){
        pa[ri]=0;
        int r2=ranges[ri]*ranges[ri];cudaMemcpyToSymbol(comm_r2,&r2,sizeof(int));
        for(int trial=0;trial<2;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){
                ss<<<32,BLK>>>(s,ri>0);
                if(ri>0)broadcast<<<32,BLK>>>();
                do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[ri]+=(float)t/NA;
        }
        pa[ri]/=2;
    }
    for(int r=0;r<8;r++){
        printf("  %3d     | %9.1f |",ranges[r],pa[r]);
        if(r==0)printf(" baseline\\n");else printf(" %.2fx\\n",pa[r]/pa[0]);
    }
    return 0;
}
