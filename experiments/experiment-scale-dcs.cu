#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define MAXA 8192
#define MAXF 1600
#define SZ 256
#define BLK 128
#define STEPS 2000
__device__ int ax[MAXA],ay[MAXA],acol[MAXA],aseed[MAXA];
__device__ int fx[MAXF],fy[MAXF],falive[MAXF];
__device__ float ftimer[MAXF];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed,int na){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed,int nf){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused,int nf){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int na,int nf,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<nf;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(use_dcs&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    printf("=== DCS at Different Scales ===\\n");
    printf("256x256, grab=12, 2000 steps\\n");
    printf("Agents | Food | NoDCS | DCS | Lift\\n");
    printf("-------------------------------\\n");
    int configs[][3]={{128,12},{512,50},{2048,200},{4096,400},{8192,800}};
    for(int ci=0;ci<5;ci++){
        int na=configs[ci][0],nf=configs[ci][1];
        int ab=(na+BLK-1)/BLK,fb=(nf+BLK-1)/BLK;
        float pn=0,pd=0;
        for(int trial=0;trial<2;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_w<<<ab,BLK>>>(42+trial,na);init_f<<<fb,BLK>>>(999+trial,nf);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<ab,BLK>>>(s,na,nf,0);do_resp<<<fb,BLK>>>(0,nf);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[MAXA];cudaMemcpyFromSymbol(hc,acol,na*sizeof(int));
            long t=0;for(int i=0;i<na;i++)t+=hc[i];pn+=(float)t/na;
            z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_w<<<ab,BLK>>>(42+trial,na);init_f<<<fb,BLK>>>(999+trial,nf);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<ab,BLK>>>(s,na,nf,1);do_resp<<<fb,BLK>>>(0,nf);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,na*sizeof(int));
            t=0;for(int i=0;i<na;i++)t+=hc[i];pd+=(float)t/na;
        }
        pn/=2;pd/=2;
        printf(" %5d | %4d | %5.0f | %3.0f | %.2fx\\n",na,nf,pn,pd,pd/pn);
    }
    return 0;
}
