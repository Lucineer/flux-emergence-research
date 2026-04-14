#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int *pfx,*pfy,*pfalive;
__device__ float *pftimer;
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
__global__ void init_f(int seed,int food){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;
    int s=seed+i*777;pfx[i]=s%SZ;pfy[i]=(s*31)%SZ;pfalive[i]=1;pftimer[i]=0;
}
__global__ void do_resp(int unused,int food){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=food)return;
    if(!pfalive[i]){pftimer[i]+=1.0f;if(pftimer[i]>50.0f){pfalive[i]=1;pftimer[i]=0;}}
}
__global__ void ss(int step,int food,int use_dcs,float gr){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=(int)(gr*gr);
    int bd=999999,bf=-1;
    for(int f=0;f<food;f++){if(!pfalive[f])continue;int d=td(ax[i],ay[i],pfx[f],pfy[f]);if(d<bd){bd=d;bf=f;}}
    if(use_dcs&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2&&pfalive[bf]){pfalive[bf]=0;acol[i]++;dcs_x[0]=pfx[bf];dcs_y[0]=pfy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(pfalive[bf]){pfalive[bf]=0;acol[i]++;dcs_x[0]=pfx[bf];dcs_y[0]=pfy[bf];dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=pfx[bf]-ax[i],dy=pfy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int maxf=800;
    int *d_fx,*d_fy,*d_fa;float *d_ft;
    cudaMalloc(&d_fx,maxf*sizeof(int));cudaMalloc(&d_fy,maxf*sizeof(int));
    cudaMalloc(&d_fa,maxf*sizeof(int));cudaMalloc(&d_ft,maxf*sizeof(float));
    cudaMemcpyToSymbol(pfx,&d_fx,sizeof(int*));
    cudaMemcpyToSymbol(pfy,&d_fy,sizeof(int*));
    cudaMemcpyToSymbol(pfalive,&d_fa,sizeof(int*));
    cudaMemcpyToSymbol(pftimer,&d_ft,sizeof(float*));
    printf("=== Scarcity x Grab x DCS ===\\n");
    printf("Food | Grab | NoDCS | DCS | Lift\\n");
    printf("------------------------------------\\n");
    int foods[]={50,100,200,400,800};
    float grabs[]={6.0f,12.0f,24.0f};
    for(int fi=0;fi<5;fi++){
        int food=foods[fi];
        int fb=(food+BLK-1)/BLK;
        for(int gi=0;gi<3;gi++){
            float gr=grabs[gi];
            float pa_nodcs=0,pa_dcs=0;
            for(int trial=0;trial<3;trial++){
                int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial,food);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,food,0,gr);do_resp<<<fb,BLK>>>(0,food);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();
                int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
                long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa_nodcs+=(float)t/NA;
                z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
                init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial,food);cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,food,1,gr);do_resp<<<fb,BLK>>>(0,food);if(s%500==0)cudaDeviceSynchronize();}
                cudaDeviceSynchronize();
                cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
                t=0;for(int i=0;i<NA;i++)t+=hc[i];pa_dcs+=(float)t/NA;
            }
            pa_nodcs/=3;pa_dcs/=3;
            printf("%4d | %4.0f | %5.0f | %3.0f | %.2fx\\n",food,gr,pa_nodcs,pa_dcs,pa_dcs/pa_nodcs);
        }
        printf("---\\n");
    }
    cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fa);cudaFree(d_ft);
    return 0;
}
