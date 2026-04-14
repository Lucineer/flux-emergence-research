#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000

// Wall map: 1=wall, 0=open
__device__ char wall[SZ*SZ];
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
__device__ int is_wall(int x,int y){return wall[to(y)*SZ+to(x)];}

__global__ void init_walls(int mode){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=SZ||y>=SZ)return;
    int idx=y*SZ+x;
    if(mode==0){wall[idx]=0;return;} // open
    if(mode==1){
        // Vertical wall at x=128, gap at y=64-192
        wall[idx]=(x==128&&!(y>=64&&y<192))?1:0;return;
    }
    if(mode==2){
        // Cross: vertical at 128 gap 64-192, horizontal at 128 gap 64-192
        wall[idx]=((x==128&&!(y>=64&&y<192))||(y==128&&!(x>=64&&x<192)))?1:0;return;
    }
    if(mode==3){
        // Grid: 4 quadrants with single-cell gaps
        wall[idx]=((x==64||x==192||y==64||y==192)&&!(x==128&&y==128))?1:0;return;
    }
    if(mode==4){
        // Maze-like: concentric rings with 4 gaps each
        wall[idx]=0;
        int cx=128,cy=128;
        int d1=(x-cx)*(x-cx)+(y-cy)*(y-cy);
        if(d1>=4096&&d1<4225)wall[idx]=1; // ring r=64
        if(d1>=1024&&d1<1089)wall[idx]=1; // ring r=32
        // Gaps at 0,90,180,270 degrees
        for(int r=0;r<2;r++){
            int rr=(r==0)?64:32;
            for(int a=0;a<4;a++){
                int gx=cx+(rr*(a%2?1:0)*(a<2?1:-1));
                int gy=cy+(rr*(a%2?0:1)*(a<2?1:-1));
                wall[((gy+SZ)%SZ)*SZ+((gx+SZ)%SZ)]=0;
            }
        }
        return;
    }
}

__global__ void init_w(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;
    // Spawn on open ground
    do{ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;}while(is_wall(ax[i],ay[i]));
    acol[i]=0;
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;
    do{fx[i]=s%SZ;fy[i]=(s*31)%SZ;}while(is_wall(fx[i],fy[i]));
    falive[i]=1;ftimer[i]=0;
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
            if(dx!=0||dy!=0){
                int nx=to(ax[i]+dx/2),ny=to(ay[i]+dy/2);
                if(!is_wall(nx,ny)){ax[i]=nx;ay[i]=ny;}
                else{
                    // Try sliding along wall
                    if(!is_wall(to(ax[i]+dx/2),ay[i]))ax[i]=to(ax[i]+dx/2);
                    else if(!is_wall(ax[i],to(ay[i]+dy/2)))ay[i]=to(ay[i]+dy/2);
                }
            }
            // Recheck nearest food at new position
            bd=999999;bf=-1;
            for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
            if(dd<=g2&&bf>=0&&falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
            return;
        }
    }
    if(bf>=0&&bd<=g2){
        if(falive[bf]){falive[bf]=0;acol[i]++;dcs_x[0]=fx[bf];dcs_y[0]=fy[bf];dcs_v[0]=1;}
    }else if(bf>=0){
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){
            int nx=to(ax[i]+dx),ny=to(ay[i]+dy);
            if(!is_wall(nx,ny)){ax[i]=nx;ay[i]=ny;}
            else if(!is_wall(to(ax[i]+dx),ay[i]))ax[i]=to(ax[i]+dx);
            else if(!is_wall(ax[i],to(ay[i]+dy)))ay[i]=to(ay[i]+dy);
        }
    }
}
int main(){
    dim3 grid(SZ/BLK,SZ/BLK);dim3 blk(BLK,BLK);
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== Barrier Topology x DCS ===\\n");
    printf("4096 agents, 400 food, grab=12\\n");
    printf("Topology          | NoDCS | DCS | Lift\\n");
    printf("------------------------------------------\\n");
    char*names[]={"Open","1-Wall+gap","Cross+gaps","Grid+gaps","Concentric"};
    for(int mode=0;mode<5;mode++){
        float pa_nodcs=0,pa_dcs=0;
        for(int trial=0;trial<3;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_walls<<<grid,blk>>>(mode);init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,0);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa_nodcs+=(float)t/NA;
            z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_walls<<<grid,blk>>>(mode);init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,1);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            t=0;for(int i=0;i<NA;i++)t+=hc[i];pa_dcs+=(float)t/NA;
        }
        pa_nodcs/=3;pa_dcs/=3;
        printf("%-18s | %5.0f | %3.0f | %.2fx\\n",names[mode],pa_nodcs,pa_dcs,pa_dcs/pa_nodcs);
    }
    return 0;
}
