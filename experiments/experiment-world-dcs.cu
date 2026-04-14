#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int td_w(int x1,int y1,int x2,int y2,int sz){
    int dx=x1-x2;if(dx<-sz/2)dx+=sz;if(dx>sz/2)dx-=sz;
    int dy=y1-y2;if(dy<-sz/2)dy+=sz;if(dy>sz/2)dy-=sz;
    return dx*dx+dy*dy;
}
__device__ int to_w(int v,int sz){return((v%sz)+sz)%sz;}

__global__ void init_w(int seed,int sz){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%sz;ay[i]=rn(&aseed[i])%sz;acol[i]=0;
}
__global__ void init_f(int seed,int sz){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%sz;fy[i]=(s*31)%sz;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step,int sz,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td_w(ax[i],ay[i],fx[f],fy[f],sz);if(d<bd){bd=d;bf=f;}}
    if(use_dcs&&dcs_v[0]){
        int dd=td_w(ax[i],ay[i],dcs_x[0],dcs_y[0],sz);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-sz/2)dx+=sz;if(dx>sz/2)dx-=sz;if(dy<-sz/2)dy+=sz;if(dy>sz/2)dy-=sz;
            if(dx!=0||dy!=0){ax[i]=to_w(ax[i]+dx/2,sz);ay[i]=to_w(ay[i]+dy/2,sz);}
            if(dd<=g2&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-sz/2)dx+=sz;if(dx>sz/2)dx-=sz;if(dy<-sz/2)dy+=sz;if(dy>sz/2)dy-=sz;
        if(dx!=0||dy!=0){ax[i]=to_w(ax[i]+dx,sz);ay[i]=to_w(ay[i]+dy,sz);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== World Size x DCS ===\\n");
    printf("Same 4096 agents, 400 food, varying world\\n");
    printf("World | Density | NoDCS | DCS | Lift\\n");
    printf("---------------------------------------\\n");
    int sizes[]={64,128,256,512};
    for(int si=0;si<4;si++){
        int sz=sizes[si];
        float density=(float)(NA+FOOD)/(float)(sz*sz);
        float pa_nodcs=0,pa_dcs=0;
        for(int trial=0;trial<3;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_w<<<32,BLK>>>(42+trial,sz);init_f<<<fb,BLK>>>(999+trial,sz);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,sz,0);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa_nodcs+=(float)t/NA;
            z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_w<<<32,BLK>>>(42+trial,sz);init_f<<<fb,BLK>>>(999+trial,sz);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,sz,1);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            t=0;for(int i=0;i<NA;i++)t+=hc[i];pa_dcs+=(float)t/NA;
        }
        pa_nodcs/=3;pa_dcs/=3;
        printf("%4d | %7.3f | %5.0f | %3.0f | %.2fx\\n",sz,density,pa_nodcs,pa_dcs,pa_dcs/pa_nodcs);
    }
    return 0;
}
