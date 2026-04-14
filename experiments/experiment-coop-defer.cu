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
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    // Cooperative deferral: check if any neighbor (in agent array) is closer
    if(mode>=1&&bf>=0){
        int defer=0;
        // Sample a few neighbors (not all 4096 — too expensive)
        for(int n=0;n<8;n++){
            int j=(i+n*512)%NA;if(j==i)continue;
            if(td(ax[j],ay[j],fx[bf],fy[bf])<bd){defer=1;break;}
        }
        if(defer&&mode==1)return; // pure defer: skip this turn
        if(defer&&mode==2){
            // defer but explore randomly
            int dx=(rn(&aseed[i])%3)-1,dy=(rn(&aseed[i])%3)-1;
            ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);return;
        }
        if(defer&&mode==3){
            // defer but go to SECOND nearest food
            int bd2=999999,bf2=-1;
            for(int f=0;f<FOOD;f++){if(!falive[f]||f==bf)continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd2){bd2=d;bf2=f;}}
            if(bf2>=0){int dx=fx[bf2]-ax[i],dy=fy[bf2]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
            return;
        }
    }
    
    if(bf>=0&&bd<=g2){if(falive[bf])falive[bf]=0;acol[i]++;}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Cooperative Deferral ===\\n");
    printf("4096 agents, 400 food, grab=12, 2000 steps\\n");
    float pa[4];
    char*names[]={"Greedy","Defer-skip","Defer-explore","Defer-second"};
    for(int mode=0;mode<4;mode++){
        pa[mode]=0;
        for(int trial=0;trial<3;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
        }
        pa[mode]/=3;
    }
    for(int m=0;m<4;m++){
        if(m==0)printf("  %-16s | %.0f/agent (baseline)\\n",names[m],pa[m]);
        else printf("  %-16s | %.0f/agent (%.2fx)\\n",names[m],pa[m],pa[m]/pa[0]);
    }
    return 0;
}
