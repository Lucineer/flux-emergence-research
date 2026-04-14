#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],aalive[NA];
__device__ float ahp[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1],dcs_owner[1];

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){
    int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    return dx*dx+dy*dy;
}
__global__ void init_w(int seed,float hp){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;
    acol[i]=0;ahp[i]=hp;aalive[i]=1;
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step,int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA||!aalive[i])return;
    float e=ahp[i];
    int g2=144;
    int bd=999999,bf=-1;
    if(e>0.002f){
        for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
        e-=0.002f;
    }
    // Mode 0: no dcs, 1: dcs persist, 2: dcs clear on owner death
    int can_dcs=0;
    if(mode>=1&&dcs_v[0]){
        if(mode==2&&!aalive[dcs_owner[0]]){dcs_v[0]=0;goto skip_dcs;}
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd)can_dcs=1;
    }
    if(can_dcs){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);e-=0.001f;}
        if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;e+=0.15f;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;dcs_owner[0]=i;}
        ahp[i]=e;return;
    }
    skip_dcs:
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;e+=0.15f;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;dcs_owner[0]=i;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);e-=0.001f;}
    }
    e-=0.001f;if(e<=0){aalive[i]=0;}ahp[i]=e;
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== DCS Owner Death ===\\n");
    printf("4096 agents, 400 food, HP=0.5 (some die)\\n");
    printf("Mode | Collection | Alive | Lift\\n");
    printf("---------------------------------\\n");
    char*names[]={"NoDCS","DCS-persist","DCS-clear-death"};
    for(int mode=0;mode<3;mode++){
        float pa=0;int al=0;
        for(int trial=0;trial<3;trial++){
            int z[3]={0,0,0};cudaMemcpyToSymbol(dcs_v,z,sizeof(int));cudaMemcpyToSymbol(dcs_owner,z+1,sizeof(int));
            init_w<<<32,BLK>>>(42+trial,0.15f);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA],ha[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);cudaMemcpyFromSymbol(ha,aalive,sizeof(int)*NA);
            long t=0;int a=0;for(int i=0;i<NA;i++){t+=hc[i];a+=ha[i];}pa+=(float)t/NA;al+=a;
        }
        pa/=3;al/=3;
        float base_pa=0;printf("%-16s| %9.1f | %4d |",names[mode],pa,al);
        if(mode==0)printf(" baseline\\n");else printf(" %.2fx\\n",pa/base_pa);
        if(mode==0)base_pa=pa;
    }
    return 0;
}
