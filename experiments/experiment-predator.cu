#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 2048
#define NPRED 64
#define FOOD 400
#define SZ 256
#define BLK 128
#define STEPS 3000

__device__ int ax[NA],ay[NA],acol[NA],aseed[NA];
__device__ int px[NPRED],py[NPRED],pseed[NPRED],pcol[NPRED];
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
__global__ void init_p(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NPRED)return;
    pseed[i]=seed+i*997;px[i]=rn(&pseed[i])%SZ;py[i]=rn(&pseed[i])%SZ;pcol[i]=0;
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
    // mode 0: nodcs nopred, 1: dcs nopred, 2: nodcs pred, 3: dcs pred, 4: dcs pred-follow-dcs
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    
    // Agent movement
    if(i<NA){
        int g2=144;
        int bd=999999,bf=-1;
        for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
        
        // Check if predator nearby (flee)
        int flee=0;
        if(mode>=2){
            for(int p=0;p<NPRED;p++){
                int d=td(ax[i],ay[i],px[p],py[p]);
                if(d<400){flee=1;break;} // flee if predator within 20 cells
            }
        }
        
        if(flee){
            // Run away from nearest predator
            int nd=999999,np=-1;
            for(int p=0;p<NPRED;p++){int d=td(ax[i],ay[i],px[p],py[p]);if(d<nd){nd=d;np=p;}}
            if(np>=0){
                int dx=ax[i]-px[np],dy=ay[i]-py[np];
                if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
                if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
            }
            return;
        }
        
        if((mode==1||mode==3||mode==4)&&dcs_v[0]){
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
    
    // Predator movement
    if(i<NPRED&&(mode>=2)){
        // Find nearest agent
        int nd=999999,ni=-1;
        for(int a=0;a<NA;a+=8){int d=td(px[i],py[i],ax[a],ay[a]);if(d<nd){nd=d;ni=a;}}
        
        // Mode 4: predators follow DCS too
        if(mode==4&&dcs_v[0]){
            int dd=td(px[i],py[i],dcs_x[0],dcs_y[0]);
            if(dd<nd){nd=dd;ni=-1;
                int dx=dcs_x[0]-px[i],dy=dcs_y[0]-py[i];
                if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
                if(dx!=0||dy!=0){px[i]=to(px[i]+dx*2);py[i]=to(py[i]+dy*2);}
            }
        }
        
        if(ni>=0&&nd<6400){ // chase within 80 cells
            int dx=ax[ni]-px[i],dy=ay[ni]-py[i];
            if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
            if(dx!=0||dy!=0){px[i]=to(px[i]+dx*2);py[i]=to(py[i]+dy*2);} // predators faster
        }
        
        // Check if caught agent
        for(int a=0;a<NA;a+=8){
            if(td(px[i],py[i],ax[a],ay[a])<9){
                // Caught! Agent respawns far away
                ax[a]=rn(&aseed[a])%SZ;ay[a]=rn(&aseed[a])%SZ;
                acol[a]=0; // lose collected food
                pcol[i]++;
            }
        }
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;int pb=(NPRED+BLK-1)/BLK;
    printf("=== Predator-Prey x DCS ===\\n");
    printf("2048 agents, 64 predators, 400 food, grab=12\\n");
    printf("Mode              | Agent coll | Pred kills\\n");
    printf("--------------------------------------------\\n");
    char*names[]={"NoDCS","DCS","Pred","DCS+Pred","DCS+Pred-DCS"};
    for(int mode=0;mode<5;mode++){
        float pa=0,pk=0;
        for(int trial=0;trial<3;trial++){
            int z[1];z[0]=0;cudaMemcpyToSymbol(dcs_v,z,sizeof(int));
            init_w<<<32,BLK>>>(42+trial);init_f<<<fb,BLK>>>(999+trial);
            if(mode>=2)init_p<<<pb,BLK>>>(77+trial);
            cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,mode);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA],hp[NPRED];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            if(mode>=2)cudaMemcpyFromSymbol(hp,pcol,sizeof(int)*NPRED);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
            long tk=0;if(mode>=2)for(int i=0;i<NPRED;i++)tk+=hp[i];pk+=tk;
        }
        pa/=3;pk/=3;
        printf("%-18s | %10.1f | %9.0f\\n",names[mode],pa,pk);
    }
    return 0;
}
