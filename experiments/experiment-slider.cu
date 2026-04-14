#include <stdio.h>
#include <cuda_runtime.h>
#define NA 256
#define FOOD 25
#define SZ 256
#define BLK 128
#define STEPS 500
#define TRIALS 1024

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int violation_pct; // 0-100: food speed = violation_pct * 3

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}

__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}

__global__ void move_food(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD||!falive[i])return;
    int spd=violation_pct*3;
    if(spd==0)return;
    int s=fx[i]*31+fy[i]*17+clock();
    fx[i]=to(fx[i]+(rn(&s)%(2*spd+1))-spd);
    fy[i]=to(fy[i]+(rn(&s)%(2*spd+1))-spd);
}

__global__ void do_resp(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>30.0f){falive[i]=1;ftimer[i]=0;}}
}

__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
            if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            bd=999999;bf=-1;
            for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            else if(bf>=0&&bd<=g2&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
        if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}

int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== DCS Condition Violation Slider ===\n");
    printf("256 agents, 25 food, grab=12, 500 steps, 1024 trials\n");
    printf("Violating: food mobility (0=static -> 100=max speed)\n\n");
    printf("Pct  | DCS_Collection | NoDCS_Collection | DCS_vs_None\n");
    printf("-----+----------------+------------------+-------------\n");

    for(int vp=0;vp<=100;vp+=5){
        cudaMemcpyToSymbol(violation_pct,&vp,sizeof(int));
        float dc=0,nc=0;
        for(int t=0;t<TRIALS;t++){
            int z=0;cudaMemcpyToSymbol(dcs_v,&z,sizeof(int));
            // DCS trial
            init_w<<<2,BLK>>>(t*31);init_f<<<fb,BLK>>>(t*47);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<2,BLK>>>(s);move_food<<<fb,BLK>>>();do_resp<<<fb,BLK>>>();if(s%200==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long total=0;for(int i=0;i<NA;i++)total+=hc[i];dc+=(float)total/NA;

            // No-DCS trial
            z=0;cudaMemcpyToSymbol(dcs_v,&z,sizeof(int));
            init_w<<<2,BLK>>>(t*31+1);init_f<<<fb,BLK>>>(t*47+1);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<2,BLK>>>(s);move_food<<<fb,BLK>>>();do_resp<<<fb,BLK>>>();if(s%200==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            total=0;for(int i=0;i<NA;i++)total+=hc[i];nc+=(float)total/NA;
        }
        dc/=TRIALS; nc/=TRIALS;
        float ratio=nc>0?dc/nc:0;
        printf("%3d%% | %14.1f | %16.1f | %11.3f\n",vp,dc,nc,ratio);
    }
    return 0;
}
