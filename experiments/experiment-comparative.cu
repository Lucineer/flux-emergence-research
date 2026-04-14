#include <stdio.h>
#include <cuda_runtime.h>
#define BLK 128
#define W 256
#define STEPS 3000

__device__ int ax[4096],ay[4096],acol[4096],aseed[4096];
__device__ int fx[400],fy[400];
__device__ unsigned int falive[400];
__device__ float ftimer[400];
__device__ int na_g, food_g;

// Stigmergy: food scent trail
__device__ float scent[256*256]; // deposited when food collected

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%W)+W)%W;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;int dy=y1-y2;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;return dx*dx+dy*dy;}

__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na_g)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%W;ay[i]=rn(&aseed[i])%W;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food_g)return;int s=seed+i*777;fx[i]=s%W;fy[i]=(s*31)%W;falive[i]=1;ftimer[i]=0;}
__global__ void clear_scent(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<W*W)return;scent[i]=0;}
__global__ void do_resp(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food_g)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>40.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void decay_scent(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*W)return;scent[i]*=0.98f;}

__global__ void ss_nodcs(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na_g)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<food_g;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=g2){unsigned int old=atomicExch(&falive[bf],0);if(old)acol[i]++;}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}

__global__ void ss_dcs(int step){
    // DCS: shared point
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na_g)return;
    extern __shared__ int sh[];
    int dcs_x=sh[0],dcs_y=sh[1],dcs_v=sh[2];
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<food_g;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(dcs_v){
        int dd=td(ax[i],ay[i],dcs_x,dcs_y);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x-ax[i],dy=dcs_y-ay[i];
            if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            bd=999999;bf=-1;
            for(int f=0;f<food_g;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
            if(bf>=0&&bd<=g2){unsigned int old=atomicExch(&falive[bf],0);if(old){acol[i]++;sh[0]=fx[bf];sh[1]=fy[bf];sh[2]=1;}}
            return;
        }
    }
    if(bf>=0&&bd<=g2){unsigned int old=atomicExch(&falive[bf],0);if(old){acol[i]++;sh[0]=fx[bf];sh[1]=fy[bf];sh[2]=1;}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}

__global__ void ss_stig(int step){
    // Stigmergy: follow scent gradient + deposit on collection
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na_g)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<food_g;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    // Also check scent
    int sd=999999,sx=-1,sy=-1;
    for(int dy=-6;dy<=6;dy++)for(int dx=-6;dx<=6;dx++){
        int nx=to(ax[i]+dx),ny=to(ay[i]+dy);
        if(scent[ny*W+nx]>0.1f){int d=dx*dx+dy*dy;if(d<sd){sd=d;sx=nx;sy=ny;}}
    }
    // Prefer food over scent
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
        }else{
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
        }
    }
}

int main(){
    int nb_max=32,fb_max=4;
    printf("=== Comparative Advantage: What The User Feels ===\n");
    printf("Measuring: tasks completed per agent per 1000 steps\n");
    printf("256x256 world, grab=12, 3000 steps, 32 trials\n\n");

    int configs[][3]={{256,26},{512,51},{1024,102},{2048,205}};
    int nc=4;

    for(int ci=0;ci<nc;ci++){
        int na=configs[ci][0],nf=configs[ci][1];
        cudaMemcpyToSymbol(na_g,&na,sizeof(int));
        cudaMemcpyToSymbol(food_g,&nf,sizeof(int));
        int nb=(na+BLK-1)/BLK,fb=(nf+BLK-1)/BLK;
        float nodcs=0,dcs=0,stig=0;
        for(int t=0;t<32;t++){
            // NoDCS
            init_w<<<nb,BLK>>>(t*31);init_f<<<fb,BLK>>>(t*47);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss_nodcs<<<nb,BLK>>>(s);do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[4096];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*na);
            long tot=0;for(int i=0;i<na;i++)tot+=hc[i];nodcs+=(float)tot/na;

            // DCS
            init_w<<<nb,BLK>>>(t*31+2);init_f<<<fb,BLK>>>(t*47+2);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss_dcs<<<nb,BLK,12>>>(s);do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*na);
            tot=0;for(int i=0;i<na;i++)tot+=hc[i];dcs+=(float)tot/na;

            // Stigmergy
            clear_scent<<<2,BLK>>>();cudaDeviceSynchronize();
            init_w<<<nb,BLK>>>(t*31+3);init_f<<<fb,BLK>>>(t*47+3);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss_stig<<<nb,BLK>>>(s);decay_scent<<<2,BLK>>>();do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*na);
            tot=0;for(int i=0;i<na;i++)tot+=hc[i];stig+=(float)tot/na;
        }
        nodcs/=32;dcs/=32;stig/=32;
        printf("--- %d agents, %d food (1:%.0f) ---\n",na,nf,(float)na/nf);
        printf("  NoDCS:     %.2f tasks/agent\n",nodcs);
        printf("  DCS:       %.2f tasks/agent  (%+.1f%% vs noDCS)\n",dcs,nodcs>0?(dcs/nodcs-1)*100:0);
        printf("  Stigmergy: %.2f tasks/agent  (%+.1f%% vs noDCS)\n",stig,nodcs>0?(stig/nodcs-1)*100:0);
        printf("  Best:      %s\n",dcs>=stig&&dcs>=nodcs?"DCS":stig>=dcs&&stig>=nodcs?"Stigmergy":"NoDCS");
        printf("  Real gain: %+.2f extra tasks/agent/day (8h, scaled to 1000 steps)\n\n",
            (dcs>=stig?dcs:stig)-nodcs);
    }
    return 0;
}
