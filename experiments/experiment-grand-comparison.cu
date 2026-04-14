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
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int mem_x[NA*MEMSIZE],mem_y[NA*MEMSIZE],mem_v[NA*MEMSIZE];
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;
    for(int m=0;m<MEMSIZE;m++){mem_x[i*MEMSIZE+m]=0;mem_y[i*MEMSIZE+m]=0;mem_v[i*MEMSIZE+m]=0;}
}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int use_mem,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    // Check memory
    if(use_mem&&bf<0){
        for(int m=0;m<MEMSIZE;m++){
            if(!mem_v[i*MEMSIZE+m])continue;
            int md=td(ax[i],ay[i],mem_x[i*MEMSIZE+m],mem_y[i*MEMSIZE+m]);
            if(md<g2*4){
                int dx=mem_x[i*MEMSIZE+m]-ax[i],dy=mem_y[i*MEMSIZE+m]-ay[i];
                if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
                if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
                if(md<=g2){
                    int found=0;
                    for(int f=0;f<FOOD;f++){if(falive[f]&&td(ax[i],ay[i],fx[f],fy[f])<=g2){found=1;falive[f]=0;acol[i]++;mem_x[i*MEMSIZE+m]=fx[f];mem_y[i*MEMSIZE+m]=fy[f];break;}}
                    if(!found)mem_v[i*MEMSIZE+m]=0;
                }
                return;
            }
        }
    }
    
    // Check DCS
    if(use_dcs&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;int sl=acol[i]%MEMSIZE;mem_x[i*MEMSIZE+sl]=fx[bf];mem_y[i*MEMSIZE+sl]=fy[bf];mem_v[i*MEMSIZE+sl]=1;}
            return;
        }
    }
    
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;int sl=acol[i]%MEMSIZE;mem_x[i*MEMSIZE+sl]=fx[bf];mem_y[i*MEMSIZE+sl]=fy[bf];mem_v[i*MEMSIZE+sl]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== GRAND COMPARISON: All Coordination Mechanisms ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n");
    printf("Mechanism       | Collection | vs None\\n");
    printf("----------------------------------------\\n");
    int modes[4][2]={{0,0},{1,0},{0,1},{1,1}};
    char*names[]={"None","Memory(4)","DCS","Memory+DCS"};
    float pa[4];
    for(int mi=0;mi<4;mi++){
        pa[mi]=0;
        for(int trial=0;trial<3;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,modes[mi][0],modes[mi][1]);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mi]+=(float)t/NA;
        }
        pa[mi]/=3;
    }
    for(int m=0;m<4;m++){
        if(m==0)printf("  %-16s | %10.1f | baseline\\n",names[m],pa[m]);
        else printf("  %-16s | %10.1f | +%.0f%%\\n",names[m],pa[m],100.0*(pa[m]/pa[0]-1));
    }
    printf("\\nMemory alone: +%.0f%%\\n",100.0*(pa[1]/pa[0]-1));
    printf("DCS alone: +%.0f%%\\n",100.0*(pa[2]/pa[0]-1));
    printf("Memory+DCS: +%.0f%% (stacking: %.2fx)\\n",100.0*(pa[3]/pa[0]-1),(pa[3]/pa[0])/((pa[1]/pa[0]-1)+(pa[2]/pa[0]-1)+1));
    return 0;
}
