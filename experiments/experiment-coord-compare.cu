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
__device__ int perc_r2;
__device__ int heat[SZ*SZ]; // stigmergy heat map
__device__ int heat_deposit;
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void init_heat(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=SZ*SZ)return;heat[i]=0;}
__global__ void decay_heat(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=SZ*SZ)return;if(heat[i]>0)heat[i]--;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int mode,int perc){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d>bd)continue;if(perc&&d>perc_r2)continue;bd=d;bf=f;}
    
    // Stigmergy: check if nearby heat suggests food direction
    int stig_dx=0,stig_dy=0;
    if(mode==2||mode==4){ // stigmergy or stig+dcs
        int best_h=0;
        for(int dx=-3;dx<=3;dx++)for(int dy=-3;dy<=3;dy++){
            int nx=to(ax[i]+dx),ny=to(ay[i]+dy);
            if(dx==0&&dy==0)continue;
            if(heat[ny*SZ+nx]>best_h){best_h=heat[ny*SZ+nx];stig_dx=dx;stig_dy=dy;}
        }
    }
    
    // DCS
    if((mode==1||mode==4)&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2){for(int f=0;f<FOOD;f++){if(falive[f]&&td(ax[i],ay[i],fx[f],fy[f])<=g2){falive[f]=0;acol[i]++;dcs_x[0]=fx[f];dcs_y[0]=fy[f];dcs_v[0]=1;heat[to(fy[f])*SZ+to(fx[f])]+=heat_deposit;break;}}}
            return;
        }
    }
    
    // Stigmergy follow
    if((mode==2||mode==4)&&stig_dx!=0&&bd>g2){
        ax[i]=to(ax[i]+stig_dx);ay[i]=to(ay[i]+stig_dy);
        // Check if we arrived at food
        for(int f=0;f<FOOD;f++){if(falive[f]&&td(ax[i],ay[i],fx[f],fy[f])<=g2){falive[f]=0;acol[i]++;dcs_x[0]=fx[f];dcs_y[0]=fy[f];dcs_v[0]=1;heat[to(fy[f])*SZ+to(fx[f])]+=heat_deposit;break;}}
        return;
    }
    
    // Random walk (mode 3)
    if(mode==3&&bd>g2){
        int dx=(rn(&aseed[i])%3)-1,dy=(rn(&aseed[i])%3)-1;
        ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);
        return;
    }
    
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;heat[to(fy[bf])*SZ+to(fx[bf])]+=heat_deposit;}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK,hs=(SZ*SZ+BLK-1)/BLK;
    printf("=== Coordination Comparison (True Limited Perc) ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps\\n");
    printf("Perc | None | DCS | Stig | Random | Stig+DCS\\n");
    printf("-------------------------------------------\\n");
    int percs[]={12,24,48};
    int hd=5;cudaMemcpyToSymbol(heat_deposit,&hd,sizeof(int));
    for(int pi=0;pi<3;pi++){
        int pr2=percs[pi]*percs[pi];cudaMemcpyToSymbol(perc_r2,&pr2,sizeof(int));
        float pa[5]={0};
        for(int mode=0;mode<5;mode++){
            for(int trial=0;trial<2;trial++){
                int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);init_heat<<<hs,BLK>>>();cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode,1);do_resp<<<fb,BLK>>>(0);if(s%100==0)decay_heat<<<hs,BLK>>>();if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();
                int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
                long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
            }
            pa[mode]/=2;
        }
        printf("  %2d | %5.0f | %3.0f | %4.0f | %5.0f | %5.0f\\n",percs[pi],pa[0],pa[1],pa[2],pa[3],pa[4]);
    }
    return 0;
}
