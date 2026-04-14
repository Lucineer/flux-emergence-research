#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],adirection[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int perc_r2;
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;adirection[i]=rn(&aseed[i])%4;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d>bd)continue;if(d>perc_r2)continue;bd=d;bf=f;}
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;}return;}
    if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}return;}
    // No food visible — explore
    int dx[]={1,0,-1,0},dy[]={0,1,0,-1};
    if(mode==0){
        // Random walk
        int rdx=(rn(&aseed[i])%3)-1,rdy=(rn(&aseed[i])%3)-1;
        ax[i]=to(ax[i]+rdx);ay[i]=to(ay[i]+rdy);
    }else if(mode==1){
        // Ballistic (persistent direction)
        ax[i]=to(ax[i]+dx[adirection[i]]);ay[i]=to(ay[i]+dy[adirection[i]]);
        if(step%300==0)adirection[i]=rn(&aseed[i])%4;
    }else if(mode==2){
        // Lévy flight (mostly short, occasionally long)
        if(rn(&aseed[i])%10==0){
            ax[i]=to(ax[i]+dx[rn(&aseed[i])%4]*8);
            ay[i]=to(ay[i]+dy[rn(&aseed[i])%4]*8);
        }else{
            ax[i]=to(ax[i]+dx[rn(&aseed[i])%4]);
            ay[i]=to(ay[i]+dy[rn(&aseed[i])%4]);
        }
    }else{
        // Spiral (gradually expand radius)
        int angle=step*3+i*1000;
        int r=2+step/200;
        ax[i]=to(ax[i]+(r%50==0)?dx[adirection[i]]:(rn(&aseed[i])%3)-1);
        ay[i]=to(ay[i]+(r%50==0)?dy[adirection[i]]:(rn(&aseed[i])%3)-1);
        if(step%50==0)adirection[i]=(adirection[i]+1)%4;
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Exploration with True Limited Perception ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n");
    printf("Perc | Random | Ballistic | Levy | Spiral\\n");
    printf("--------------------------------------\\n");
    int percs[]={12,24,48};
    char*enames[]={"Random","Ballistic","Levy","Spiral"};
    for(int pi=0;pi<3;pi++){
        int pr2=percs[pi]*percs[pi];cudaMemcpyToSymbol(perc_r2,&pr2,sizeof(int));
        float pa[4]={0};
        for(int mode=0;mode<4;mode++){
            for(int trial=0;trial<2;trial++){
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();
                int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
                long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
            }
            pa[mode]/=2;
        }
        printf("  %2d | %6.0f | %8.0f | %4.0f | %5.0f\\n",percs[pi],pa[0],pa[1],pa[2],pa[3]);
    }
    return 0;
}
