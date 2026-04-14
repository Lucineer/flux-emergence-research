#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000
#define MAXE 100
__device__ int ax[NA],ay[NA],acol[NA],aseed[NA],aenergy[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[1],dcs_y[1],dcs_v[1];
__device__ int move_cost; // energy cost per step moved
__device__ int food_energy; // energy gained per food
__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;return dx*dx+dy*dy;}
__global__ void init_w(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;acol[i]=0;aenergy[i]=MAXE;}
__global__ void init_f(int seed){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;}
__global__ void do_resp(int unused){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}}
__global__ void ss(int step,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    if(aenergy[i]<=0)return; // dead agent
    int g2=144,bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(use_dcs&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            int dist=abs(dx)+abs(dy);
            if(dist>0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);aenergy[i]-=move_cost;}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;aenergy[i]+=food_energy;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){if(falive[bf]){falive[bf]=0;acol[i]++;aenergy[i]+=food_energy;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
    else if(bf>=0){int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);aenergy[i]-=move_cost;}}
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Energy Dynamics ===\\n");
    printf("4096 agents, 400 food, grab=12, 3000 steps, max_energy=100\\n");
    printf("Cost | FoodE | NoDCS | DCS | Alive%% | DCS lift\\n");
    printf("--------------------------------------------\\n");
    int costs[]={0,1,2,5};
    int fes[]={10,20,50};
    for(int ci=0;ci<4;ci++){
        cudaMemcpyToSymbol(move_cost,&costs[ci],sizeof(int));
        for(int fi=0;fi<3;fi++){
            cudaMemcpyToSymbol(food_energy,&fes[fi],sizeof(int));
            float pn=0,pd=0,an=0,ad=0;
            for(int trial=0;trial<2;trial++){
                int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,0);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();int hc[NA],he[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);cudaMemcpyFromSymbol(he,aenergy,sizeof(int)*NA);
                long t=0,alive=0;for(int i=0;i<NA;i++){t+=hc[i];if(he[i]>0)alive++;}pn+=(float)t/NA;an+=(float)alive/NA;
                z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,1);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);cudaMemcpyFromSymbol(he,aenergy,sizeof(int)*NA);
                t=0;alive=0;for(int i=0;i<NA;i++){t+=hc[i];if(he[i]>0)alive++;}pd+=(float)t/NA;ad+=(float)alive/NA;
            }
            pn/=2;pd/=2;an/=2;ad/=2;
            printf("  %2d  |  %2d   | %5.0f | %3.0f | %5.1f%% | %.2fx\\n",costs[ci],fes[fi],pn,pd,100*ad,pd/pn);
        }
    }
    return 0;
}
