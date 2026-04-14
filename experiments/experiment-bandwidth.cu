// experiment-bandwidth.cu
// Communication Bandwidth Limits
//
// Law 3 says "information only matters under scarcity."
// What if we limit HOW MUCH agents can communicate per step?
//
// Hypothesis: bandwidth-limited agents outperform unlimited because
// they're forced to share only high-value information.
//
// Scenarios:
// 0. No communication (control) — agents act independently
// 1. Unlimited gossip — share everything with nearby agents
// 2. Bandwidth-limited — share top-N most valuable observations
// 3. Broadcast-only — agents broadcast position but not observations

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID 128
#define NA 1024
#define NR 200
#define STEPS 3000
#define TRIALS 5

struct A { float x,y,spd; int col; float best_dx,best_dy,best_d2; };
struct R { float x,y; int active; };

__device__ int d_tot;
__device__ float rng(unsigned int *s){*s=*s*1103515245u+12345u;return(float)(*s&0x7FFFFFFFu)/(float)0x7FFFFFFFu;}

__global__ void rst(){if(threadIdx.x==0&&blockIdx.x==0)d_tot=0;}
__global__ void iA(A *a,int n,unsigned int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int r=s+i*7919;
    a[i].x=rng(&r)*GRID;a[i].y=rng(&r)*GRID;a[i].spd=3.0f+rng(&r)*2.0f;a[i].col=0;
    a[i].best_dx=0;a[i].best_dy=0;a[i].best_d2=999999.0f;}
__global__ void iR(R *r,int n,unsigned int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int r2=s+i*65537;
    r[i].x=rng(&r2)*GRID;r[i].y=rng(&r2)*GRID;r[i].active=1;}

// MODE 0: No communication — random walk + grab
__global__ void step_none(A *a,int na,R *r,int nr,unsigned int s,int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;unsigned int r2=s+i*131+step*997;
    a[i].x=fmodf(a[i].x+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    a[i].y=fmodf(a[i].y+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    float g2=9.0f;
    for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;
    if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    if(dx*dx+dy*dy<g2){r[j].active=0;a[i].col++;atomicAdd(&d_tot,1);break;}}}

// MODE 1: Perception + directed movement (agents sense food and move toward nearest)
__global__ void step_perceive(A *a,int na,R *r,int nr,unsigned int s,int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;unsigned int r2=s+i*131+step*997;
    
    // Find nearest food
    float bd=400.0f;float bx=0,by=0;int found=0;
    for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=r[j].x-a[i].x,dy=r[j].y-a[i].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;
    if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    float d2=dx*dx+dy*dy;if(d2<bd){bd=d2;bx=dx;by=dy;found=1;}}
    
    if(found&&bd<400.0f){
        float d=sqrtf(bd);if(d>0.1f){a[i].x=fmodf(a[i].x+bx/d*a[i].spd+GRID,GRID);
        a[i].y=fmodf(a[i].y+by/d*a[i].spd+GRID,GRID);}
    } else {
        a[i].x=fmodf(a[i].x+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
        a[i].y=fmodf(a[i].y+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    }
    
    float g2=9.0f;
    for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;
    if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    if(dx*dx+dy*dy<g2){r[j].active=0;a[i].col++;atomicAdd(&d_tot,1);break;}}}

// MODE 2: Gossip — agents share food locations with nearby agents
// Each agent can receive up to bandwidth_limit shared food locations
__global__ void step_gossip(A *a,int na,R *r,int nr,unsigned int s,int step,int bw_limit){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;unsigned int r2=s+i*131+step*997;
    
    // Each agent scans food and nearby agents
    // First: find my own nearest food
    float my_bd=400.0f;float my_bx=0,my_by=0;
    for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=r[j].x-a[i].x,dy=r[j].y-a[i].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;
    if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    float d2=dx*dx+dy*dy;if(d2<my_bd){my_bd=d2;my_bx=dx;my_by=dy;}}
    
    // Move toward my nearest OR a gossiped location (if closer)
    float tgt_dx=my_bx,tgt_dy=my_by,tgt_d2=my_bd;
    int gossip_used=0;
    
    // "Gossip": check a few random other agents' nearest food (simulated)
    // In real implementation this would use shared memory
    for(int g=0;g<bw_limit;g++){
        int other=(i*7+g*31+step*13)%na;
        float ox=a[other].x,oy=a[other].y;
        // Check food near that agent
        for(int j=0;j<nr;j++){if(!r[j].active)continue;
        float dx=r[j].x-a[i].x,dy=r[j].y-a[i].y;
        if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;
        if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
        float d2=dx*dx+dy*dy;if(d2<tgt_d2){tgt_d2=d2;tgt_dx=dx;tgt_dy=dy;gossip_used++;break;}}
    }
    
    if(tgt_d2<400.0f){float d=sqrtf(tgt_d2);if(d>0.1f){
    a[i].x=fmodf(a[i].x+tgt_dx/d*a[i].spd+GRID,GRID);
    a[i].y=fmodf(a[i].y+tgt_dy/d*a[i].spd+GRID,GRID);}}
    else{a[i].x=fmodf(a[i].x+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    a[i].y=fmodf(a[i].y+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);}
    
    float g2=9.0f;
    for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;
    if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    if(dx*dx+dy*dy<g2){r[j].active=0;a[i].col++;atomicAdd(&d_tot,1);break;}}}

__global__ void rsp(R *r,int n,unsigned int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||r[i].active)return;unsigned int r2=s+i*3571;
    r[i].x=rng(&r2)*GRID;r[i].y=rng(&r2)*GRID;r[i].active=1;}

int main(){
    printf("=== Communication Bandwidth Limits ===\n");
    printf("Agents:%d Resources:%d Steps:%d Trials:%d\n\n",NA,NR,STEPS,TRIALS);
    int bs=256,ag=(NA+bs-1)/bs,rg=(NR+bs-1)/bs;
    A *da;R *dr;cudaMalloc(&da,NA*sizeof(A));cudaMalloc(&dr,NR*sizeof(R));
    srand(time(NULL));

    // Baseline comparison
    printf("Mode                Total     PerAgent  vs None\n");
    printf("------------------------------------------------\n");
    
    int none_tot=0;
    for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000;
    rst<<<1,1>>>();iA<<<ag,bs>>>(da,NA,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
    for(int step=0;step<STEPS;step++){step_none<<<ag,bs>>>(da,NA,dr,NR,s,step);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
    int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));none_tot+=h;}
    printf("%-20s%-10d%-10.2f1.00x\n","No communication",none_tot/TRIALS,(float)(none_tot/TRIALS)/NA);

    int perc_tot=0;
    for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+11111;
    rst<<<1,1>>>();iA<<<ag,bs>>>(da,NA,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
    for(int step=0;step<STEPS;step++){step_perceive<<<ag,bs>>>(da,NA,dr,NR,s,step);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
    int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));perc_tot+=h;}
    printf("%-20s%-10d%-10.2f%.2fx\n","Perception only",perc_tot/TRIALS,(float)(perc_tot/TRIALS)/NA,(float)perc_tot/none_tot);

    // Bandwidth sweep
    int bws[]={1,2,3,5,8,13,21,34};
    for(int b=0;b<8;b++){
        int tot=0;
        for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+bws[b]*1000;
        rst<<<1,1>>>();iA<<<ag,bs>>>(da,NA,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
        for(int step=0;step<STEPS;step++){step_gossip<<<ag,bs>>>(da,NA,dr,NR,s,step,bws[b]);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
        int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));tot+=h;}
        printf("Gossip bw=%-12d%-10d%-10.2f%.2fx\n",bws[b],tot/TRIALS,(float)(tot/TRIALS)/NA,(float)tot/none_tot);
    }

    // Agent density sweep at optimal bandwidth
    printf("\nAgent Density Sweep (perception only)\n");
    printf("Agents  Total     PerAgent  vs None@1024\n");
    int ns[]={128,256,512,1024,2048};
    for(int ni=0;ni<5;ni++){
        int n=ns[ni],nag=(n+bs-1)/bs;A *da2;cudaMalloc(&da2,n*sizeof(A));
        int nt=0,nn=0;
        for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+ni*10000;
        rst<<<1,1>>>();iA<<<nag,bs>>>(da2,n,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
        for(int step=0;step<STEPS;step++){step_none<<<nag,bs>>>(da2,n,dr,NR,s,step);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
        int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));nn+=h;}
        for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+ni*10000+5555;
        rst<<<1,1>>>();iA<<<nag,bs>>>(da2,n,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
        for(int step=0;step<STEPS;step++){step_perceive<<<nag,bs>>>(da2,n,dr,NR,s,step);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
        int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));nt+=h;}
        printf("%-8d%-10d%-10.2f%.2fx\n",n,nt/TRIALS,(float)(nt/TRIALS)/n,(float)(nt/TRIALS)/(none_tot/TRIALS));
        cudaFree(da2);
    }

    cudaFree(da);cudaFree(dr);return 0;
}
