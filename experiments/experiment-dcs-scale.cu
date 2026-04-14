#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define SZ 256
#define BLK 128
#define STEPS 3000
#define GUILDS 8

__device__ int ax[65536],ay[65536],acol[65536],aseed[65536];
__device__ int fx[3200],fy[3200],falive[3200];
__device__ float ftimer[3200];
__device__ int dcs_x[GUILDS],dcs_y[GUILDS],dcs_v[GUILDS];
__device__ float gr=12.0f;
__device__ int NA_GLOBAL,FOOD_GLOBAL;

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){
    int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    return dx*dx+dy*dy;
}

__global__ void init_w(int na,int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;
}
__global__ void init_f(int food,int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int food){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int na,int food,int step,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=na)return;
    int g=i%GUILDS;
    int bd=999999,bf=-1;
    for(int f=0;f<food;f++){
        if(!falive[f])continue;
        int d=td(ax[i],ay[i],fx[f],fy[f]);
        if(d<bd){bd=d;bf=f;}
    }
    if(use_dcs&&dcs_v[g]){
        int dd=td(ax[i],ay[i],dcs_x[g],dcs_y[g]);
        if(dd<(int)(gr*gr*4)&&bd>dd){
            int dx=dcs_x[g]-ax[i],dy=dcs_y[g]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
            if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=(int)(gr*gr)&&falive[bf]){
                falive[bf]=0;acol[i]++;dcs_x[g]=fx[bf];dcs_y[g]=fy[bf];dcs_v[g]=1;
            }
            return;
        }
    }
    if(bf>=0&&bd<=(int)(gr*gr)){
        if(falive[bf]){falive[bf]=0;acol[i]++;int g=i%GUILDS;
        dcs_x[g]=fx[bf];dcs_y[g]=fy[bf];dcs_v[g]=1;}
    } else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
        if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int configs[][3]={{512,50,0},{512,50,1},{512,200,0},{512,200,1},
        {2048,200,0},{2048,200,1},{2048,800,0},{2048,800,1},
        {4096,400,0},{4096,400,1},{4096,100,0},{4096,100,1},
        {8192,800,0},{8192,800,1},{16384,1600,0},{16384,1600,1}};
    int nc=16;
    printf("=== DCS Ring Buffer Scale Sweep ===\n");
    printf("Agents x Food x DCS | Total | PerAgent\n");
    printf("--------------------------------------\n");
    for(int c=0;c<nc;c++){
        int na=configs[c][0],food=configs[c][1],dcs=configs[c][2];
        int bk=(na+BLK-1)/BLK,fb=(food+BLK-1)/BLK;
        int z[GUILDS];for(int i=0;i<GUILDS;i++)z[i]=0;
        cudaMemcpyToSymbol(dcs_v,z,sizeof(int)*GUILDS);
        init_w<<<bk,BLK>>>(na,42+c);
        init_f<<<fb,BLK>>>(food,999+c);
        cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){
            ss<<<bk,BLK>>>(na,food,s,dcs);
            do_resp<<<fb,BLK>>>(food);
            if(s%500==0)cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        int hc[65536];
        cudaMemcpyFromSymbol(hc,acol,sizeof(int)*na);
        long total=0;for(int i=0;i<na;i++)total+=hc[i];
        printf("%5d x %4d x %d | %6ld | %7.1f\n",na,food,dcs,total,(float)total/na);
    }
    return 0;
}
