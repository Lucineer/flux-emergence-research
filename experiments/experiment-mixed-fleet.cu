#include <stdio.h>
#include <cuda_runtime.h>
#define BLK 128
#define W 256
#define STEPS 2000
#define NA 512
#define NF 51

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],arole[NA];
__device__ int fx[NF],fy[NF];
__device__ unsigned int falive[NF];
__device__ float ftimer[NF];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%W)+W)%W;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;int dy=y1-y2;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;return dx*dx+dy*dy;}

__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%W;ay[i]=rn(&aseed[i])%W;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;int s=seed+i*777;fx[i]=s%W;fy[i]=(s*31)%W;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>40.0f){falive[i]=1;ftimer[i]=0;}}}

__global__ void step_mixed(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int role=arole[i],g2=144,bd=999999,bf=-1;
    for(int f=0;f<NF;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    if(role==0){
        // EXPLORER: random walk + collect if close
        if(bf>=0&&bd<=g2){unsigned int o=atomicExch(&falive[bf],0);if(o)acol[i]++;}
        else{int dx=rn(&aseed[i])%7-3,dy=rn(&aseed[i])%7-3;if(dx||dy){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
    }else if(role==1){
        // EXPLOITER: DCS + greedy
        if(dcs_v[0]){
            int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
            if(dd<g2*4&&bd>dd){
                int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
                if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;
                if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
                bd=999999;bf=-1;
                for(int f=0;f<NF;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
                if(bf>=0&&bd<=g2){unsigned int o=atomicExch(&falive[bf],0);if(o){acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
                return;
            }
        }
        if(bf>=0&&bd<=g2){unsigned int o=atomicExch(&falive[bf],0);if(o){acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
        else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
    }else{
        // DEFENDER: stay near spawn (128,128), collect only if very close
        int home_d=td(ax[i],ay[i],128,128);
        if(bf>=0&&bd<=36){unsigned int o=atomicExch(&falive[bf],0);if(o)acol[i]++;}
        else if(home_d>400){
            // Move toward home
            int dx=128-ax[i],dy=128-ay[i];
            if(dx<-W/2)dx+=W;if(dx>W/2)dx-=W;if(dy<-W/2)dy+=W;if(dy>W/2)dy-=W;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
        }
    }
}

int main(){
    printf("=== Mixed Fleet Composition ===\n");
    printf("512 agents, 51 food, 256x256, 2000 steps, 32 trials\n");
    printf("Explorer=random walk, Exploiter=DCS, Defender=home guard\n\n");
    printf("Exp:Exp:Def | Total tasks | Per-agent | Best role\n");
    printf("------------+-------------+-----------+----------\n");

    int nb=(NA+BLK-1)/BLK,fb=(NF+BLK-1)/BLK;
    
    int ratios[][3]={{100,0,0},{0,100,0},{0,0,100},{50,50,0},{50,0,50},{0,50,50},
                     {33,33,34},{20,60,20},{60,20,20},{10,80,10},{80,10,10}};
    int nr=sizeof(ratios)/sizeof(ratios[0]);
    
    for(int ri=0;ri<nr;ri++){
        int pct_e=ratios[ri][0],pct_x=ratios[ri][1],pct_d=ratios[ri][2];
        int ne=NA*pct_e/100,nx=NA*pct_x/100,nd=NA-ne-nx;
        float total=0;
        
        for(int t=0;t<32;t++){
            // Set roles
            int roles[NA];int z=0;
            cudaMemcpyToSymbol(dcs_v,&z,sizeof(int));
            for(int i=0;i<ne;i++)roles[i]=0;
            for(int i=ne;i<ne+nx;i++)roles[i]=1;
            for(int i=ne+nx;i<NA;i++)roles[i]=2;
            cudaMemcpyToSymbol(arole,roles,sizeof(int)*NA);
            
            init_w<<<nb,BLK>>>(t*31);init_f<<<fb,BLK>>>(t*47);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){step_mixed<<<nb,BLK>>>();do_resp<<<fb,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long tot=0;for(int i=0;i<NA;i++)tot+=hc[i];total+=(float)tot;
        }
        total/=32;
        
        // Get per-role stats for last trial
        int roles_h[NA],hc_h[NA];
        cudaMemcpyFromSymbol(arole,roles_h,sizeof(int)*NA);
        cudaMemcpyFromSymbol(acol,hc_h,sizeof(int)*NA);
        float re=0,rx=0,rd=0;int ce=0,cx=0,cd=0;
        for(int i=0;i<NA;i++){
            if(roles_h[i]==0){re+=hc_h[i];ce++;}
            else if(roles_h[i]==1){rx+=hc_h[i];cx++;}
            else{rd+=hc_h[i];cd++;}
        }
        
        printf("%3d:%3d:%3d  | %11.0f | %9.2f | E%.1f X%.1f D%.1f\n",
            pct_e,pct_x,pct_d,total,total/NA,
            ce?re/ce:0,cx?rx/cx:0,cd?rd/cd:0);
    }
    return 0;
}
