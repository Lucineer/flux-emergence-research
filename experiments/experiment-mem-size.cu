#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 2000
#define MAXMEM 16
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int mem_x[NA*MAXMEM],mem_y[NA*MAXMEM],mem_v[NA*MAXMEM];
__device__ int mem_size;
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;
    for(int m=0;m<MAXMEM;m++){mem_x[i*MAXMEM+m]=0;mem_y[i*MAXMEM+m]=0;mem_v[i*MAXMEM+m]=0;}
}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,ms=mem_size,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf<0&&ms>0){
        for(int m=0;m<ms;m++){
            if(!mem_v[i*MAXMEM+m])continue;
            int md=td(ax[i],ay[i],mem_x[i*MAXMEM+m],mem_y[i*MAXMEM+m]);
            if(md<g2*4){
                int dx=mem_x[i*MAXMEM+m]-ax[i],dy=mem_y[i*MAXMEM+m]-ay[i];
                if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
                if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
                if(md<=g2){int found=0;for(int f=0;f<FOOD;f++){if(falive[f]&&td(ax[i],ay[i],fx[f],fy[f])<=g2){found=1;falive[f]=0;acol[i]++;mem_x[i*MAXMEM+m]=fx[f];mem_y[i*MAXMEM+m]=fy[f];break;}}if(!found)mem_v[i*MAXMEM+m]=0;}
                return;
            }
        }
    }
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;if(ms>0){int sl=acol[i]%ms;mem_x[i*MAXMEM+sl]=fx[bf];mem_y[i*MAXMEM+sl]=fy[bf];mem_v[i*MAXMEM+sl]=1;}}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Memory Size Scaling ===\\n");
    printf("4096 agents, 400 food, grab=12, 2000 steps\\n");
    printf("Slots | Collection | vs None\\n");
    printf("-----------------------------\\n");
    int sizes[]={0,1,2,4,8,16};
    float base=0;
    for(int si=0;si<6;si++){
        cudaMemcpyToSymbol(mem_size,&sizes[si],sizeof(int));
        float pa=0;
        for(int trial=0;trial<2;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
        }
        pa/=2;
        if(si==0)base=pa;
        printf("  %2d   | %9.1f |",sizes[si],pa);
        if(si==0)printf(" baseline\\n");else printf(" %.2fx\\n",pa/base);
    }
    return 0;
}
