#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 200
#define SZ 256
#define BLK 128
#define STEPS 3000
#define GUILDS 1

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[GUILDS],dcs_y[GUILDS],dcs_step[GUILDS],dcs_v[GUILDS];
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
__global__ void ss(int food,int step,int mode,int ttl){
    // mode: 0=no dcs, 1=dcs no inval, 2=dcs+ttl inval, 3=dcs+age decay
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int bd=999999,bf=-1;
    for(int f=0;f<food;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    int use_dcs=0;
    if(mode>=1&&dcs_v[0]){
        int age=step-dcs_step[0];
        if(mode==1)use_dcs=1;
        else if(mode==2&&age<ttl)use_dcs=1;
        else if(mode==3){
            float weight=max(0.0f,1.0f-(float)age/(float)ttl);
            if(weight>0.3f)use_dcs=1;
        }
    }
    if(use_dcs){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<(int)(gr*gr*4)&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=(int)(gr*gr)&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_step[0]=step;dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=(int)(gr*gr)){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_step[0]=step;dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== DCS Invalidation + Moving Food ===\n");
    printf("4096 agents, 200 food, migration speed=2\n");
    printf("Mode              | TTL | Collection/agent | vs NoDCS\n");
    printf("-------------------------------------------------------\n");
    int speeds[]={0,1,2,4};char*sn[]={"static","slow","medium","fast"};
    int ttls[]={5,10,20,50,100,999};
    for(int si=0;si<4;si++){
        for(int mi=0;mi<4;mi++){
            int ttl=mi==0?999:mi==1?20:mi==2?10:5;
            float pa=0;
            for(int trial=0;trial<3;trial++){
                int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(FOOD,s,mi,ttl);migrate_f<<<fb,BLK>>>(FOOD,speeds[si]);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();
                int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
                long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
            }
            pa/=3;
            char*mn[]={"NoDCS","DCS-no-inval","DCS-TTL20","DCS-TTL10","DCS-TTL5"};
            printf("%-16s | %3d | %15.1f |\n",mn[mi],ttl,pa);
        }
        printf("---\n");
    }
    return 0;
}
