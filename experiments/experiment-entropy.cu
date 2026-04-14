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
__global__ void ss(int step,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int g2=144;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(use_dcs&&dcs_v[0]){
        int dd=td(ax[i],ay[i],dcs_x[0],dcs_y[0]);
        if(dd<g2*4&&bd>dd){
            int dx=dcs_x[0]-ax[i],dy=dcs_y[0]-ay[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== DCS Information Content ===\\n");
    printf("Tracking DCS point changes over 3000 steps\\n\\n");
    int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
    init_w<<<32,BLK>>>(42);init_f<<<fb,BLK>>>(999);cudaDeviceSynchronize();
    int prev_x=-1,prev_y=-1;
    int changes=0,repeated=0,unique_x[256],unique_y[256];
    int spatial_reuse=0; // DCS points that are close to previous
    for(int x=0;x<256;x++){unique_x[x]=0;unique_y[x]=0;}
    int dist_sum=0,dist_count=0;
    for(int s=0;s<STEPS;s++){
        ss<<<32,BLK>>>(s,1);do_resp<<<fb,BLK>>>(0);cudaDeviceSynchronize();
        int hx,hy;
        cudaMemcpyFromSymbol(&hx,dcs_x,sizeof(int));
        cudaMemcpyFromSymbol(&hy,dcs_y,sizeof(int));
        if(hx!=prev_x||hy!=prev_y){
            changes++;
            if(prev_x>=0){
                int dx=hx-prev_x;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
                int dy=hy-prev_y;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
                int d=dx*dx+dy*dy;
                dist_sum+=d;dist_count++;
                if(d<256)spatial_reuse++; // within 16 cells of previous
            }
            unique_x[hx]++;unique_y[hy]++;
            prev_x=hx;prev_y=hy;
        }else{repeated++;}
    }
    printf("DCS point changes: %d of %d steps (%.1f%%)\\n",changes,STEPS,100.0*changes/STEPS);
    printf("Unchanged steps: %d\\n",repeated);
    printf("Avg jump distance: %.1f cells\\n",(float)sqrt((float)dist_sum/dist_count));
    printf("Spatial reuse (< 16 cells): %d (%.1f%% of changes)\\n",spatial_reuse,100.0*spatial_reuse/changes);
    // Spatial entropy of DCS x-coordinates
    double entropy=0;
    for(int x=0;x<256;x++){if(unique_x[x]>0){double p=(double)unique_x[x]/changes;entropy-=p*log2(p);}}
    printf("Spatial entropy (x): %.2f bits (max %.2f for 256 values)\\n",entropy,log2(256));
    double yent=0;
    for(int y=0;y<256;y++){if(unique_y[y]>0){double p=(double)unique_y[y]/changes;yent-=p*log2(p);}}
    printf("Spatial entropy (y): %.2f bits\\n",yent);
    printf("Combined entropy: %.2f bits (max 16.0 for uniform 256x256)\\n",entropy+yent);
    printf("\\nInterpretation:\\n");
    printf("DCS updates every %.1f steps on average\\n",(float)STEPS/changes);
    printf("Each update carries %.2f bits of spatial information\\n",entropy+yent);
    printf("DCS throughput: %.2f bits/step\\n",(entropy+yent)*changes/STEPS);
    return 0;
}
