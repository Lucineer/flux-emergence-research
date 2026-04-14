// experiment-phase-transition.cu
// Phase Transition — finding the exact scarcity threshold
// where adaptive behavior flips from helpful to harmful.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID 256
#define STEPS 2000
#define TRIALS 3

struct A { float x,y,grab,base_grab,speed,col; int ld; };
struct F { float x,y; int active; };

__device__ int d_tot;
__device__ float lcg(unsigned int *s) { *s=*s*1103515245u+12345u; return (float)(*s&0x7FFFFFFFu)/(float)0x7FFFFFFFu; }

__global__ void reset() { if(threadIdx.x==0&&blockIdx.x==0) d_tot=0; }
__global__ void initA(A *a,int n,unsigned int s) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; unsigned int r=s+i*7919; a[i].x=lcg(&r)*GRID; a[i].y=lcg(&r)*GRID; a[i].base_grab=2.5f; a[i].grab=2.5f; a[i].speed=3.0f; a[i].col=0; a[i].ld=0; }
__global__ void initF(F *f,int n,unsigned int s) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; unsigned int r=s+i*65537; f[i].x=lcg(&r)*GRID; f[i].y=lcg(&r)*GRID; f[i].active=1; }

__global__ void step_fixed(A *a,int na,F *f,int nf,unsigned int s,int step) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=na) return;
    unsigned int r=s+i*131+step*997;
    a[i].x=fmodf(a[i].x+(lcg(&r)-0.5f)*a[i].speed*2+GRID,GRID);
    a[i].y=fmodf(a[i].y+(lcg(&r)-0.5f)*a[i].speed*2+GRID,GRID);
    float g2=a[i].grab*a[i].grab;
    for(int j=0;j<nf;j++){ if(!f[j].active)continue; float dx=a[i].x-f[j].x,dy=a[i].y-f[j].y; if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID; if(dx*dx+dy*dy<g2){f[j].active=0;a[i].col++;atomicAdd(&d_tot,1);}}
}

__global__ void step_adapt(A *a,int na,F *f,int nf,unsigned int s,int step,float threshold) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=na) return;
    unsigned int r=s+i*131+step*997;
    // Count local food
    int lc=0; float perc=20.0f;
    for(int j=0;j<nf;j++){ if(!f[j].active)continue; float dx=a[i].x-f[j].x,dy=a[i].y-f[j].y; if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID; if(dx*dx+dy*dy<perc*perc)lc++;}
    a[i].ld=lc;
    // Adapt grab range based on local density vs threshold
    float target;
    if(lc < (int)(threshold * 0.5f)) target = a[i].base_grab * 2.0f;  // scarce: expand
    else if(lc > threshold) target = a[i].base_grab * 0.7f;  // abundant: contract
    else target = a[i].base_grab;
    a[i].grab += (target - a[i].grab) * 0.1f;
    a[i].grab = fmaxf(1.0f, fminf(a[i].grab, 8.0f));
    // Move toward food if scarce
    float dx=0,dy=0;
    if(lc < (int)(threshold * 0.7f)) {
        float bd=perc*perc;
        for(int j=0;j<nf;j++){if(!f[j].active)continue;float fdx=f[j].x-a[i].x,fdy=f[j].y-a[i].y;if(fdx>GRID*0.5f)fdx-=GRID;if(fdx<-GRID*0.5f)fdx+=GRID;if(fdy>GRID*0.5f)fdy-=GRID;if(fdy<-GRID*0.5f)fdy+=GRID;float d2=fdx*fdx+fdy*fdy;if(d2<bd){bd=d2;dx=fdx;dy=fdy;}}
        if(bd<perc*perc){float d=sqrtf(bd);dx=dx/d*a[i].speed;dy=dy/d*a[i].speed;}else{dx=(lcg(&r)-0.5f)*a[i].speed*2;dy=(lcg(&r)-0.5f)*a[i].speed*2;}
    } else { dx=(lcg(&r)-0.5f)*a[i].speed*2; dy=(lcg(&r)-0.5f)*a[i].speed*2; }
    a[i].x=fmodf(a[i].x+dx+GRID,GRID); a[i].y=fmodf(a[i].y+dy+GRID,GRID);
    float g2=a[i].grab*a[i].grab;
    for(int j=0;j<nf;j++){if(!f[j].active)continue;float fdx=a[i].x-f[j].x,fdy=a[i].y-f[j].y;if(fdx>GRID*0.5f)fdx-=GRID;if(fdx<-GRID*0.5f)fdx+=GRID;if(fdy>GRID*0.5f)fdy-=GRID;if(fdy<-GRID*0.5f)fdy+=GRID;if(fdx*fdx+fdy*fdy<g2){f[j].active=0;a[i].col++;atomicAdd(&d_tot,1);}}
}

__global__ void respawn(F *f,int n,unsigned int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||f[i].active)return;unsigned int r=s+i*3571;f[i].active=1;f[i].x=lcg(&r)*GRID;f[i].y=lcg(&r)*GRID;}

int main() {
    printf("=== Phase Transition: Adaptive vs Fixed Across Scarcity Gradient ===\n\n");
    
    int bs=256, na=2048, ag=(na+bs-1)/bs;
    A *da; F *df; cudaMalloc(&da,na*sizeof(A)); cudaMalloc(&df,800*sizeof(F));
    srand(time(NULL));
    
    // Fine-grained sweep of food/agent ratio
    printf("Food/Agent  Fixed       Adaptive    Lift        Phase\n");
    printf("--------------------------------------------------------\n");
    
    for (int food_per_agent_x10 = 2; food_per_agent_x10 <= 80; food_per_agent_x10 += 2) {
        int nf = na * food_per_agent_x10 / 10;
        if (nf > 800) nf = 800;
        int nfg = (nf+bs-1)/bs;
        F *df2; cudaMalloc(&df2, nf*sizeof(F));
        
        int fixed_t=0, adapt_t=0;
        for (int t=0; t<TRIALS; t++) {
            unsigned int s=(unsigned int)time(NULL)+t*50000+food_per_agent_x10*10000;
            // Fixed
            reset<<<1,1>>>();initA<<<ag,bs>>>(da,na,s);initF<<<nfg,bs>>>(df2,nf,s+1);cudaDeviceSynchronize();
            for(int step=0;step<STEPS;step++){step_fixed<<<ag,bs>>>(da,na,df2,nf,s,step);respawn<<<nfg,bs>>>(df2,nf,s+step);cudaDeviceSynchronize();}
            int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));fixed_t+=h;
            // Adaptive (threshold=5 local food)
            reset<<<1,1>>>();initA<<<ag,bs>>>(da,na,s+999);initF<<<nfg,bs>>>(df2,nf,s+1000);cudaDeviceSynchronize();
            for(int step=0;step<STEPS;step++){step_adapt<<<ag,bs>>>(da,na,df2,nf,s+999,step,5.0f);respawn<<<nfg,bs>>>(df2,nf,s+999+step);cudaDeviceSynchronize();}
            cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));adapt_t+=h;
        }
        float lift=(float)adapt_t/fixed_t;
        const char *phase = lift > 1.1f ? "ADAPTIVE WINS" : lift < 0.9f ? "FIXED WINS" : "NEUTRAL";
        printf("0.%-8d %-12d %-12d %-12.2fx %s\n", food_per_agent_x10, fixed_t/TRIALS, adapt_t/TRIALS, lift, phase);
        cudaFree(df2);
    }
    
    // Optimal threshold sweep
    printf("\n=== Optimal Adaptation Threshold (food=200) ===\n");
    printf("Threshold   Total       vs Fixed\n");
    printf("----------------------------------\n");
    
    int nf=200, nfg=(nf+bs-1)/bs;
    F *df3; cudaMalloc(&df3,nf*sizeof(F));
    
    // Get fixed baseline
    int fixed_base=0;
    for(int t=0;t<TRIALS;t++){
        unsigned int s=(unsigned int)time(NULL)+t*50000;
        reset<<<1,1>>>();initA<<<ag,bs>>>(da,na,s);initF<<<nfg,bs>>>(df3,nf,s+1);cudaDeviceSynchronize();
        for(int step=0;step<STEPS;step++){step_fixed<<<ag,bs>>>(da,na,df3,nf,s,step);respawn<<<nfg,bs>>>(df3,nf,s+step);cudaDeviceSynchronize();}
        int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));fixed_base+=h;
    }
    printf("Fixed:      %-12d (baseline)\n", fixed_base/TRIALS);
    
    float thresholds[] = {1,2,3,4,5,6,7,8,10,15,20};
    for(int ti=0;ti<11;ti++){
        int tot=0;
        for(int t=0;t<TRIALS;t++){
            unsigned int s=(unsigned int)time(NULL)+t*50000+ti*100000;
            reset<<<1,1>>>();initA<<<ag,bs>>>(da,na,s);initF<<<nfg,bs>>>(df3,nf,s+1);cudaDeviceSynchronize();
            for(int step=0;step<STEPS;step++){step_adapt<<<ag,bs>>>(da,na,df3,nf,s,step,thresholds[ti]);respawn<<<nfg,bs>>>(df3,nf,s+step);cudaDeviceSynchronize();}
            int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));tot+=h;
        }
        printf("%-12.0f %-12d %.2fx\n", thresholds[ti], tot/TRIALS, (float)tot/fixed_base);
    }
    
    cudaFree(da);cudaFree(df);cudaFree(df3);
    return 0;
}
