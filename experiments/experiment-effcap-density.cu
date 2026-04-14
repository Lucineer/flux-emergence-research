#include <stdio.h>
#include <cuda_runtime.h>
#define BLK 128
#define STEPS 2000
#define W 256
#define NA 1024

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[800],fy[800];
__device__ unsigned int falive[800];
__device__ float ftimer[800];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int food_g;

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%W)+W)%W;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;int dy=y1-y2;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;return dx*dx+dy*dy;}

__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%W;ay[i]=rn(&aseed[i])%W;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food_g)return;int s=seed+i*777;fx[i]=s%W;fy[i]=(s*31)%W;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food_g)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>30.0f){falive[i]=1;ftimer[i]=0;}}}

__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<food_g;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            bd=999999;bf=-1;
            for(int f=0;f<food_g;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
            if(bf>=0&&bd<=g2){unsigned int old=atomicExch(&falive[bf],0);if(old){acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
            return;
        }
    }
    if(bf>=0&&bd<=g2){unsigned int old=atomicExch(&falive[bf],0);if(old){acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}

int main(){
    printf("=== Efficiency vs Food Density (1024 agents, 256x256) ===\n");
    printf("grab=12, 2000 steps, 32 trials\n\n");
    printf("Food | Agt/Food | DCS/agt | NoDCS | Lift | Eff%%\n");
    printf("-----+----------+---------+-------+------+------\n");

    int nb=(NA+BLK-1)/BLK;
    int food_counts[]={10,25,50,100,200,400,800};
    for(int fi=0;fi<7;fi++){
        int nf=food_counts[fi];
        cudaMemcpyToSymbol(food_g,&nf,sizeof(int));
        int fb=(nf+BLK-1)/BLK;
        float dc=0,nc=0;
        for(int t=0;t<32;t++){
            int z=0;cudaMemcpyToSymbol(dcs_v,&z,sizeof(int));
            init_w<<<nb,BLK>>>(t*31);init_f<<<fb,BLK>>>(t*47);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<nb,BLK>>>(s);do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long total=0;for(int i=0;i<NA;i++)total+=hc[i];dc+=(float)total/NA;

            z=0;cudaMemcpyToSymbol(dcs_v,&z,sizeof(int));
            init_w<<<nb,BLK>>>(t*31+1);init_f<<<fb,BLK>>>(t*47+1);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<nb,BLK>>>(s);do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            total=0;for(int i=0;i<NA;i++)total+=hc[i];nc+=(float)total/NA;
        }
        dc/=32;nc/=32;
        float lift=nc>0?(dc/nc-1)*100:0;
        float theo=(float)nf*66.0f/(float)NA;
        float eff=theo>0?(nc/theo)*100:0;
        printf("%4d | %8.1f | %7.2f | %5.2f | %+5.1f%% | %4.1f%%\n",
            nf,(float)NA/nf,dc,nc,lift,eff);
    }
    return 0;
}
