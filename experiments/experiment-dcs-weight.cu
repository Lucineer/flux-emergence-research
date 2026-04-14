#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
#define GUILDS 8

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[GUILDS],dcs_y[GUILDS],dcs_v[GUILDS];

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
__global__ void ss(int step,int mode){
    // mode 0: no dcs, 1: own guild, 2: nearest guild point, 3: all guilds nearest, 4: weighted avg direction
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g=i%GUILDS;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    int use_dcs=0,ddx=0,ddy=0;
    if(mode==1&&dcs_v[g]){
        int dd=td(ax[i],ay[i],dcs_x[g],dcs_y[g]);
        if(dd<g2*4&&bd>dd){ddx=dcs_x[g]-ax[i];ddy=dcs_y[g]-ay[i];use_dcs=1;}
    }else if(mode>=2){
        int best_dd=999999,best_gg=-1;
        int check_max=(mode==2)?1:GUILDS;
        for(int gg=0;gg<check_max;gg++){
            if(!dcs_v[gg])continue;
            int dd=td(ax[i],ay[i],dcs_x[gg],dcs_y[gg]);
            if(dd<best_dd){best_dd=dd;best_gg=gg;}
        }
        if(best_gg>=0&&best_dd<g2*4&&bd>best_dd){
            if(mode==4){
                // Weighted: sum directions weighted by inverse distance
                long wx=0,wy=0,wt=0;
                for(int gg=0;gg<GUILDS;gg++){
                    if(!dcs_v[gg])continue;
                    int dd=td(ax[i],ay[i],dcs_x[gg],dcs_y[gg]);
                    if(dd<g2*4&&dd>0){int w=1000000/dd;wx+=w*(dcs_x[gg]-ax[i]);wy+=w*(dcs_y[gg]-ay[i]);wt+=w;}
                }
                if(wt>0){ddx=(int)(wx/wt);ddy=(int)(wy/wt);use_dcs=1;}
            }else{
                ddx=dcs_x[best_gg]-ax[i];ddy=dcs_y[best_gg]-ay[i];use_dcs=1;
            }
        }
    }
    
    if(use_dcs&&(ddx!=0||ddy!=0)){
        if(ddx<-SZ/2)ddx+=SZ;if(ddx>SZ/2)ddx-=SZ;if(ddy<-SZ/2)ddy+=SZ;if(ddy>SZ/2)ddy-=SZ;
        ax[i]=to(ax[i]+ddx/2);ay[i]=to(ay[i]+ddy/2);
        int dd=td(ax[i],ay[i],dcs_x[g],dcs_y[g]); // check grab at current pos
        // Try grab nearest food at new position
        int gbd=999999,gbf=-1;
        for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<gbd){gbd=d;gbf=f;}}
        if(gbd<=g2&&gbf>=0&&falive[gbf]){falive[gbf]=0;acol[i]++;dcs_x[g]=fx[gbf];dcs_y[g]=fy[gbf];dcs_v[g]=1;}
        return;
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[g]=fx[bf];dcs_y[g]=fy[bf];dcs_v[g]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== DCS Multi-Point Strategies ===\\n");
    printf("4096 agents, 400 food, 8 guilds\\n");
    printf("Mode                | Collection | Lift\\n");
    printf("--------------------------------------\\n");
    char*names[]={"NoDCS","Own-guild","Nearest-1","Nearest-all","Weighted-avg"};
    float pa[5]={0};
    for(int mode=0;mode<5;mode++){
        for(int trial=0;trial<3;trial++){
            int z[GUILDS];for(int j=0;j<GUILDS;j++)z[j]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int)*GUILDS);
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
        }
        pa[mode]/=3;
    }
    for(int m=0;m<5;m++){
        if(m==0)printf("%-20s| %10.1f | baseline\\n",names[m],pa[m]);
        else printf("%-20s| %10.1f | %.2fx\\n",names[m],pa[m],pa[m]/pa[0]);
    }
    return 0;
}
