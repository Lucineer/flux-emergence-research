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
__device__ float gr=12.0f;

// LCG (Linear Congruential Generator) - same as existing
__device__ int lcg(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

// Xorshift32
__device__ int xor32(int*s){
    unsigned int x=*s;
    x^=x<<13;x^=x>>17;x^=x<<5;*s=(int)x;return(int)x;
}

// Multiply-with-carry
__device__ int mwc(int*s){
    static const unsigned int A=12345;
    unsigned int x=*s;
    *s=(int)(A*x+1);return(int)(*s>>16);
}

__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){
    int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    return dx*dx+dy*dy;
}

__global__ void init_w(int seed,int prng){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=aseed[i]%SZ;ay[i]=(aseed[i]*31)%SZ;acol[i]=0;
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__global__ void ss(int step,int prng){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    int bd=999999,bf=-1;
    for(int f=0;f<FOOD;f++){if(!falive[f])continue;int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}
    if(bf>=0&&bd<=(int)(gr*gr)){
        if(falive[bf]){falive[bf]=0;acol[i]++;}
    }else if(bf>=0){
        int r;
        if(prng==0)r=lcg(&aseed[i]);
        else if(prng==1)r=xor32(&aseed[i]);
        else r=mwc(&aseed[i]);
        int dx=fx[bf]-ax[i],dy=fy[bf]-ay[i];
        if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
        if(dx!=0||dy!=0){
            // Add randomness to movement (stochastic path)
            int jitter=r%7-3;
            ax[i]=to(ax[i]+dx+jitter);ay[i]=to(ay[i]+dy+jitter);
        }
    }
}
int main(){
    int fb=(FOOD+BLK-1)/BLK;
    printf("=== PRNG Quality Test ===\n");
    printf("4096 agents, 400 food, stochastic movement\n");
    printf("PRNG | Collection/agent (avg 5 seeds)\n");
    printf("---------------------------------------\n");
    char*names[]={"LCG","Xorshift32","MWC"};
    for(int p=0;p<3;p++){
        float pa=0;
        for(int trial=0;trial<5;trial++){
            init_w<<<32,BLK>>>(42+trial*1000,p);init_f<<<fb,BLK>>>(999+trial);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){ss<<<32,BLK>>>(s,p);do_resp<<<fb,BLK>>>(0);if(s%500==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hc[NA];cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
            long t=0;for(int i=0;i<NA;i++)t+=hc[i];pa+=(float)t/NA;
        }
        pa/=5;
        printf("%-10s | %.1f\n",names[p],pa);
    }
    printf("\nIf all PRNGs give similar results: PRNG choice does not matter\n");
    printf("If significant variance: PRNG quality affects simulation validity\n");
    return 0;
}
