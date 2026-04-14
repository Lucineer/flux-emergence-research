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
    // mode 0: no dcs no repel, 1: dcs no repel, 2: no dcs repel, 3: dcs repel
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    
    int can_dcs=0;
    if((mode==1||mode==3)&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            can_dcs=1;
        }
    }
    if(!can_dcs){
        if(bf>=0&&bd<=g2){
            if(falive[bf]){falive[bf]=0;acol[i]++;if(mode==1||mode==3){dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}}
        }else if(bf>=0){
            int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
        }
    }
    
    // Repulsion: sample nearby agents and push away
    if(mode==2||mode==3){
        int rx=0,ry=0;
        // Sample 8 neighbors by agent index proximity
        for(int j=1;j<=8;j++){
            int ni=(i+j)%NA;
            int dx=ax[i]-ax[ni];if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
            int dy=ay[i]-ay[ni];if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            int d2=dx*dx+dy*dy;
            if(d2<16&&d2>0){rx+=dx;ry+=dy;} // repel if within distance 4
        }
        if(rx!=0||ry!=0){
            // Average and apply gentle push
            ax[i]=to(ax[i]+rx/16);ay[i]=to(ay[i]+ry/16);
        }
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Agent Repulsion x DCS ===\\n");
    printf("4096 agents, 400 food, grab=12\\n");
    printf("Mode              | Collection | vs baseline\\n");
    printf("-------------------------------------------\\n");
    char*names[]={"NoDCS-NoRepel","DCS-NoRepel","NoDCS-Repel","DCS-Repel"};
    float pa[4];
    for(int mode=0;mode<4;mode++){
        pa[mode]=0;
        for(int trial=0;trial<3;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa[mode]+=(float)t/NA;
        }
        pa[mode]/=3;
    }
    printf("Baseline: %.1f\\n",pa[0]);
    for(int m=0;m<4;m++){
        printf("  %-16s | %10.1f | %.2fx\\n",names[m],pa[m],pa[m]/pa[0]);
    }
    printf("\\nRepel effect on NoDCS: %.2fx\\n",pa[2]/pa[0]);
    printf("Repel effect on DCS:   %.2fx\\n",pa[3]/pa[1]);
    printf("DCS+Repel vs NoDCS+Repel: %.2fx\\n",pa[3]/pa[2]);
    return 0;
}
