#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];

// Stigmergy: heat map of recent collections
__device__ int heat[SZ*SZ]; // collection activity per cell

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
__global__ void init_heat(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<SZ*SZ)heat[i]=0;}
__global__ void decay_heat(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<SZ*SZ&&heat[i]>0){heat[i]-=1;if(heat[i]<0)heat[i]=0;}
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step,int mode){
    // mode 0: no coord, 1: DCS, 2: stigmergy (follow heat), 3: anti-stigmergy (avoid heat)
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    if(mode==1&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;heat[to(fy[bf])*SZ+to(fx[bf])]+=10;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    
    if(mode==2||mode==3){
        // Scan nearby cells for heat gradient
        int bx=0,by=0,sign=(mode==2)?1:-1;
        for(int dx=-3;dx<=3;dx+=3){
            for(int dy=-3;dy<=3;dy+=3){
                int nx=to(ax[i]+dx),ny=to(ay[i]+dy);
                int h=heat[ny*SZ+nx];
                bx+=sign*h*dx;by+=sign*h*dy;
            }
        }
        if(bx!=0||by!=0){ax[i]=to(ax[i]+bx/20);ay[i]=to(ay[i]+by/20);}
    }
    
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;heat[to(fy[bf])*SZ+to(fx[bf])]+=10;if(mode==1){dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;int hb=(SZ*SZ+BLK-1)/BLK;
    printf("=== Stigmergy vs DCS vs None ===\\n");
    printf("4096 agents, 400 food, grab=12\\n");
    printf("Coordination     | Collection | vs None\\n");
    printf("--------------------------------------\\n");
    char*names[]={"None","DCS","Stigmergy","Anti-Stig"};
    for(int mode=0;mode<4;mode++){
        float pa=0;
        for(int trial=0;trial<3;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_heat<<<hb,BLK>>>();
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%10==0)decay_heat<<<hb,BLK>>>();cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
        }
        pa/=3;
        float base_pa=0;printf("  %-16s | %10.1f |",names[mode],pa);
        if(mode==0)printf(" baseline\\n");else printf(" %.2fx\\n",pa/base_pa);
        if(mode==0)base_pa=pa;
    }
    return 0;
}
