#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int *pfx,*pfy,*pfa;
__device__ float *pft;

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
__global__ void init_f(int seed,int food){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;
    int s=seed+i*777;pfx[i]=s%SZ;pfy[i]=(s*31)%SZ;pfa[i]=1;pft[i]=0;
}
__global__ void do_resp(int unused,int food){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;
    if(!pfa[i]){pft[i]+=1.0f;if(pft[i]>50.0f){pfa[i]=1;pft[i]=0;}}
}
__global__ void ss(int step,int food,float gr){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=(int)(gr*gr);
    int bd=999999,bf=-1;
    for(int f=0;f<food;f++){if(!pfa[f])continue;int d=td(ax[i],ay[i],pfx[f],pfy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){
        if(pfa[bf]){pfa[bf]=0;acol[i]++;}
    }else if(bf>=0){
        int dx=pfx[bf]-ax[i],dy=pfy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int maxf=800;
    int *d_fx,*d_fy,*d_fa;float *d_ft;
    cudaMalloc(&d_fx,maxf*sizeof(int));cudaMalloc(&d_fy,maxf*sizeof(int));
    cudaMalloc(&d_fa,maxf*sizeof(int));cudaMalloc(&d_ft,maxf*sizeof(float));
    cudaMemcpyToSymbol(pfx,&d_fx,sizeof(int*));
    cudaMemcpyToSymbol(pfy,&d_fy,sizeof(int*));
    cudaMemcpyToSymbol(pfa,&d_fa,sizeof(int*));
    cudaMemcpyToSymbol(pft,&d_ft,sizeof(float*));
    printf("=== Perception x Density (NoDCS baseline) ===\\n");
    printf("Food | Grab=6 | Grab=12 | Grab=24\\n");
    printf("----------------------------------\\n");
    int foods[]={50,100,200,400,800};
    float grabs[]={6.0f,12.0f,24.0f};
    for(int fi=0;fi<5;fi++){
        int food=foods[fi];int fb=(food+BLK-1)/BLK;
        printf(" %3d  |",food);
        for(int gi=0;gi<3;gi++){
            float pa=0;
            for(int trial=0;trial<3;trial++){
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial,food);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,food,grabs[gi]);do_resp<<<fb,BLK>>>(0,food);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();
                int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
                long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
            }
            pa/=3;printf(" %6.0f |",pa);
        }
        printf("\\n");
    }
    cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);cudaFree(d_ft);
    return 0;
}
