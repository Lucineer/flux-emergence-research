#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int step_col[STEPS]; // collection per step (atomic)

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
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(use_dcs&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;atomicAdd(&step_col[step],1);dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;atomicAdd(&step_col[step],1);dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Temporal Collection Patterns ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n");
    printf("Looking for oscillations, waves, bursts...\\n\\n");
    for(int mode=0;mode<2;mode++){
        int hc[STEPS];for(int i=0;i<STEPS;i++)hc[i]=0;
        cudaMemcpyToSymbol(step_col,hc,sizeof(int)*STEPS);
        int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
        init_w<<<32,BLK>>>(42);init_f<<<fb,BLK>>>(999);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);cudaDeviceSynchronize();}
        cudaMemcpyFromSymbol(hc,step_col,sizeof(int)*STEPS);
        long total=0;for(int i=0;i<STEPS;i++)total+=hc[i];
        float avg=(float)total/STEPS;
        // Compute variance and autocorrelation at lag 50
        float var=0;for(int i=0;i<STEPS;i++){float d=hc[i]-avg;var+=d*d;}var/=STEPS;
        float autocorr=0;int lag=50;
        for(int i=0;i<STEPS-lag;i++){float d1=hc[i]-avg;float d2=hc[i+lag]-avg;autocorr+=d1*d2;}
        autocorr/=(STEPS-lag);autocorr/=var;
        // Find max and min windows (100-step windows)
        int maxw=0,minw=99999,maxs=0,mins=0;
        for(int w=0;w<STEPS-100;w++){int sum=0;for(int j=0;j<100;j++)sum+=hc[w+j];if(sum>maxw){maxw=sum;maxs=w;}if(sum<minw){minw=sum;mins=w;}}
        printf("Mode %s: total=%ld, avg=%.1f, var=%.1f, autocorr(lag50)=%.3f\\n",mode?"DCS":"NoDCS",total,avg,var,autocorr);
        printf("  Max window at step %d: %d cols/100 steps (%.1f/step)\\n",maxs,maxw,maxw/100.0);
        printf("  Min window at step %d: %d cols/100 steps (%.1f/step)\\n",mins,minw,minw/100.0);
        printf("  Burst ratio: %.2fx\\n\\n",(float)maxw/minw);
    }
    return 0;
}
