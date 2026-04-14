#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 200
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_vx[1],dcs_vy[1],dcs_v[1];
__device__ float gr=12.0f;

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
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void migrate_f(int food,int speed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food||!falive[i])return;
    fx[i]=to(fx[i]+speed);fy[i]=to(fy[i]+speed);
}
__global__ void ss(int food,int step,int mode,int speed){
    // mode 0: no dcs, 1: dcs position, 2: dcs velocity (predict), 3: dcs velocity+verify
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int bd=999999,bf=-1;
    for(int f=0;f<food;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(mode>=1&&dcs_v[0]){
        int pred_x=dcs_x[0],pred_y=dcs_y[0];
        if(mode>=2){pred_x=to(pred_x+dcs_vx[0]);pred_y=to(pred_y+dcs_vy[0]);}
        int dd=td(ax[i],ay[i],pred_x,pred_y);
        if(dd<(int)(gr*gr*4)&&bd>dd){
            if(mode==3){
                // Verify: check food near predicted position
                int vbd=999999,vbf=-1;
                for(int f=0;f<food;f++){if(!falive[f])continue;
                int d=td(pred_x,pred_y,fx[f],fy[f]);if(d<vbd){vbd=d;vbf=f;}}
                if(vbf<0||vbd>(int)(gr*gr*4))goto skip_dcs;
                // Move toward verified food
                int dx2=fx[vbf]-ax[i],dy2=fy[vbf]-ay[i];
                if(dx2<-SZ/2)dx2+=SZ;if(dx2>SZ/2)dx2-=SZ;
                if(dy2<-SZ/2)dy2+=SZ;if(dy2>SZ/2)dy2-=SZ;
                if(dx2!=0||dy2!=0){ax[i]=to(ax[i]+dx2);ay[i]=to(ay[i]+dy2);}
                int gd=td(ax[i],ay[i],fx[vbf],fy[vbf]);
                if(gd<=(int)(gr*gr)&&falive[vbf]){falive[vbf]=0;acol[i]++;dcs_x[0]=fx[vbf];dcs_y[0]=fy[vbf];dcs_vx[0]=speed;dcs_vy[0]=speed;dcs_v[0]=1;}
                return;
            }
            int dx=pred_x-ax[i],dy=pred_y-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=(int)(gr*gr)&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_vx[0]=speed;dcs_vy[0]=speed;dcs_v[0]=1;}
            return;
        }
    }
    skip_dcs:
    if(bf>=0&&bd<=(int)(gr*gr)){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_vx[0]=speed;dcs_vy[0]=speed;dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== DCS Velocity Prediction ===\n");
    printf("4096 agents, 200 food, diagonal migration\n");
    printf("Speed | NoDCS | Pos-DCS | Vel-DCS | Vel+Verify\n");
    printf("-----------------------------------------------\n");
    int speeds[]={0,1,2,4};
    for(int si=0;si<4;si++){
        float pa[4]={0};
        for(int mode=0;mode<4;mode++){
            for(int trial=0;trial<3;trial++){
                int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(FOOD,s,mode,speeds[si]);migrate_f<<<fb,BLK>>>(FOOD,speeds[si]);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();
                int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
                long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
            }
            pa[mode]/=3;
        }
        printf("  %d   | %5.0f | %7.0f | %7.0f | %10.0f\n",speeds[si],pa[0],pa[1],pa[2],pa[3]);
    }
    return 0;
}
