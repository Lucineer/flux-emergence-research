#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 2000
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],ascore[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD],fval[FOOD];
__device__ float ftimer[FOOD];
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;ascore[i]=0;}
__global__ void init_f(int seed,int mode){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;if(mode==0)fval[i]=1;else if(mode==1)fval[i]=(rn(&s)%10)+1;else fval[i]=(i%20==0)?100:1;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1,best_val=0;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d>bd)continue;if(mode>=2&&fval[f]<best_val)continue;bd=d;bf=f;best_val=fval[f];}
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;ascore[i]+=fval[bf];}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Food Value Strategy ===\\n");
    printf("4096 agents, 400 food, grab=12, 2000 steps\\n");
    float pc[3],ps[3];
    char*vnames[]={"Uniform-1","Varied-1-10","Jackpot-100"};
    for(int vm=0;vm<3;vm++){
        pc[vm]=0;ps[vm]=0;
        for(int trial=0;trial<2;trial++){
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial,vm);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,vm>=1?1:0);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA],hs[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);cudaMemcpyFromSymbol(hs,ascore,sizeof(int)*NA);
            long tc=0,ts=0;for(int i=0;i<NA;i++){tc+=hc[i];ts+=hs[i];}pc[vm]+=(float)tc/NA;ps[vm]+=(float)ts/NA;
        }
        pc[vm]/=2;ps[vm]/=2;
    }
    for(int v=0;v<3;v++)printf("  %-16s | coll=%.0f | score=%.0f | avg_val=%.1f\\n",vnames[v],pc[v],ps[v],ps[v]/pc[v]);
    return 0;
}
