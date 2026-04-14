#include <stdio.h>
#include <cuda_runtime.h>
#define BLK 128
#define W 256
#define STEPS 2000
#define NA 512
#define NF 51

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[NF],fy[NF];
__device__ unsigned int falive[NF];
__device__ float ftimer[NF];
__device__ float scent[W*W];
__device__ float decay_g;

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%W)+W)%W;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;int dy=y1-y2;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;return dx*dx+dy*dy;}

__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%W;ay[i]=rn(&aseed[i])%W;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;int s=seed+i*777;fx[i]=s%W;fy[i]=(s*31)%W;falive[i]=1;ftimer[i]=0;}
__global__ void clear_scent(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<W*W)return;scent[i]=0;}
__global__ void do_resp(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>40.0f){falive[i]=1;ftimer[i]=0;}}}

__global__ void step(int step_n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<NF;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    // Follow scent gradient
    int sd=999999,sx=-1,sy=-1;
    for(int dy=-6;dy<=6;dy++)for(int dx=-6;dx<=6;dx++){
        int nx=to(ax[i]+dx),ny=to(ay[i]+dy);
        if(scent[ny*W+nx]>0.1f){int d=dx*dx+dy*dy;if(d<sd){sd=d;sx=nx;sy=ny;}}
    }
    int tx=-1,ty=-1;
    if(bf>=0&&bd<=g2){tx=fx[bf];ty=fy[bf];}
    else if(sx>=0){tx=sx;ty=sy;}
    else if(bf>=0){tx=fx[bf];ty=fy[bf];}
    if(tx>=0){
        int dx=tx-ax[i],dy=ty-ay[i];
        if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;
        if(bf>=0&&bd<=g2){
            unsigned int old=atomicExch(&falive[bf],0);
            if(old){acol[i]++;scent[to(ay[i])*W+to(ax[i])]+=5.0f;}
        }else{if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
    }
}

__global__ void decay_scent(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*W)return;scent[i]*=decay_g;}

int main(){
    printf("=== Scent Decay Optimization ===\n");
    printf("512 agents, 51 food, 256x256, 2000 steps, 32 trials\n\n");
    printf("Decay | Tasks/agent | vs NoDCS\n");
    printf("------+-------------+----------\n");

    int nb=(NA+BLK-1)/BLK,fb=(NF+BLK-1)/BLK;

    // First get NoDCS baseline
    float nodcs=0;
    for(int t=0;t<32;t++){
        init_w<<<nb,BLK>>>(t*31);init_f<<<fb,BLK>>>(t*47);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){
            int g2=144,bd=999999,bf=-1;
            // Inline noDCS step - just use the stigmergy kernel but with no scent
            // Actually just reuse step with scent=0
            step<<<nb,BLK>>>(s);
            do_resp<<<fb,BLK>>>();
            if(s%500==0)cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
        long tot=0;for(int i=0;i<NA;i++)tot+=hc[i];nodcs+=(float)tot/NA;
    }
    nodcs/=32;
    printf("NoDCS baseline: %.2f\n\n",nodcs);

    float decays[]={0.90,0.92,0.94,0.96,0.97,0.98,0.99,0.995,0.999,1.0};
    for(int di=0;di<10;di++){
        float d=decays[di];
        cudaMemcpyToSymbol(decay_g,&d,sizeof(float));
        float total=0;
        for(int t=0;t<32;t++){
            clear_scent<<<4,BLK>>>();cudaDeviceSynchronize();
            init_w<<<nb,BLK>>>(t*31+2);init_f<<<fb,BLK>>>(t*47+2);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){step<<<nb,BLK>>>(s);decay_scent<<<4,BLK>>>();do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long tot=0;for(int i=0;i<NA;i++)tot+=hc[i];total+=(float)tot/NA;
        }
        total/=32;
        printf(" %.3f |    %7.2f  | %+.1f%%\n",d,total,nodcs>0?(total/nodcs-1)*100:0);
    }
    return 0;
}
