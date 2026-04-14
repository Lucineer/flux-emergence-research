#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
#define MEMSIZE 4

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int mem_x[NA*MEMSIZE],mem_y[NA*MEMSIZE],mem_v[NA*MEMSIZE];

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
    for(int m=0;m<MEMSIZE;m++){mem_x[i*MEMSIZE+m]=0;mem_y[i*MEMSIZE+m]=0;mem_v[i*MEMSIZE+m]=0;}
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step,int mode){
    // mode 0: no memory, 1: personal memory, 2: personal memory + forget stale
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    // Check personal memory for closer target
    if(mode>=1){
        for(int m=0;m<MEMSIZE;m++){
            if(!mem_v[i*MEMSIZE+m])continue;
            int md=td(ax[i],ay[i],mem_x[i*MEMSIZE+m],mem_y[i*MEMSIZE+m]);
            if(md<bd){bd=md;bf=-1; // go to memory, not current food
                int dx=mem_x[i*MEMSIZE+m]-ax[i],dy=mem_y[i*MEMSIZE+m]-ay[i];
                if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
                if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
                // Check if we arrived
                if(md<=g2){
                    // Check for food at memory location (might be collected by others)
                    int found=0;
                    for(int f=0;f<FOOD;f++){if(falive[f]&&td(ax[i],ay[i],fx[f],fy[f])<=g2){found=1;falive[f]=0;acol[i]++;mem_x[i*MEMSIZE+m]=fx[f];mem_y[i*MEMSIZE+m]=fy[f];break;}}
                    if(!found){
                        // Memory stale - clear it
                        mem_v[i*MEMSIZE+m]=0;
                    }
                }
                return;
            }
        }
    }
    
    if(bf>=0&&bd<=g2){
        if(falive[bf]){
            falive[bf]=0;acol[i]++;
            if(mode>=1){
                // Store in memory (ring buffer)
                int slot=acol[i]%MEMSIZE;
                mem_x[i*MEMSIZE+slot]=fx[bf];mem_y[i*MEMSIZE+slot]=fy[bf];mem_v[i*MEMSIZE+slot]=1;
            }
        }
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Personal Memory vs No Memory ===\\n");
    printf("4096 agents, 400 food, grab=12, mem=4 slots\\n");
    float pa[2]={0};
    for(int mode=0;mode<2;mode++){
        for(int trial=0;trial<3;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
        }
        pa[mode]/=3;
    }
    printf("No memory:    %.0f/agent\\n",pa[0]);
    printf("Mem(4 slots): %.0f/agent (%.2fx)\\n",pa[1],pa[1]/pa[0]);
    return 0;
}
