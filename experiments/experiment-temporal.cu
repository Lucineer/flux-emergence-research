// experiment-temporal.cu — Temporal Coordination
// Hookers mark, launchers collect. Tests sequential vs simultaneous.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID 128
#define NA 512
#define NR 150
#define STEPS 3000
#define TRIALS 5

struct A { float x,y; int type,col; float spd; };
struct R { float x,y; int active,marked,mt; };

__device__ int d_tot,d_dcy;
__device__ float rng(unsigned int *s){*s=*s*1103515245u+12345u;return(float)(*s&0x7FFFFFFFu)/(float)0x7FFFFFFFu;}

__global__ void rst(){if(threadIdx.x==0&&blockIdx.x==0){d_tot=0;d_dcy=0;}}
__global__ void iA(A *a,int n,float hr,unsigned int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int r=s+i*7919;
    a[i].x=rng(&r)*GRID;a[i].y=rng(&r)*GRID;a[i].type=(i<(int)(n*hr))?0:1;a[i].col=0;a[i].spd=3.0f+rng(&r)*2.0f;}
__global__ void iR(R *r,int n,unsigned int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int r2=s+i*65537;
    r[i].x=rng(&r2)*GRID;r[i].y=rng(&r2)*GRID;r[i].active=1;r[i].marked=0;r[i].mt=-1;}

__global__ void sim(A *a,int na,R *r,int nr,unsigned int s,int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;unsigned int r2=s+i*131+step*997;
    a[i].x=fmodf(a[i].x+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    a[i].y=fmodf(a[i].y+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    for(int j=0;j<nr;j++){if(!r[j].active)continue;float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    if(dx*dx+dy*dy<9.0f){r[j].active=0;a[i].col++;atomicAdd(&d_tot,1);break;}}}

__global__ void seq(A *a,int na,R *r,int nr,unsigned int s,int step,int md,int dc){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;unsigned int r2=s+i*131+step*997;
    a[i].x=fmodf(a[i].x+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    a[i].y=fmodf(a[i].y+(rng(&r2)-0.5f)*a[i].spd*2+GRID,GRID);
    if(a[i].type==0){for(int j=0;j<nr;j++){if(!r[j].active||r[j].marked)continue;
    float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    if(dx*dx+dy*dy<9.0f){r[j].marked=1;r[j].mt=step;break;}}
    }else{for(int j=0;j<nr;j++){if(!r[j].active||r[j].marked!=1)continue;
    if(step-r[j].mt<md)continue;if(step-r[j].mt>dc){r[j].active=0;r[j].marked=0;atomicAdd(&d_dcy,1);continue;}
    float dx=a[i].x-r[j].x,dy=a[i].y-r[j].y;
    if(dx>GRID*0.5f)dx-=GRID;if(dx<-GRID*0.5f)dx+=GRID;if(dy>GRID*0.5f)dy-=GRID;if(dy<-GRID*0.5f)dy+=GRID;
    if(dx*dx+dy*dy<9.0f){r[j].active=0;r[j].marked=2;a[i].col++;atomicAdd(&d_tot,1);break;}}}}

__global__ void rsp(R *r,int n,unsigned int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||r[i].active)return;unsigned int r2=s+i*3571;
    r[i].x=rng(&r2)*GRID;r[i].y=rng(&r2)*GRID;r[i].active=1;r[i].marked=0;r[i].mt=-1;}

void run_sim(A *da,R *dr,int bs,int ag,int rg,int *out){
    int tot=0;for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+99999;
    rst<<<1,1>>>();iA<<<ag,bs>>>(da,NA,0.5f,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
    for(int step=0;step<STEPS;step++){sim<<<ag,bs>>>(da,NA,dr,NR,s,step);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
    int h;cudaMemcpyFromSymbol(&h,d_tot,sizeof(int));tot+=h;}*out=tot/TRIALS;}

void run_seq(A *da,R *dr,int bs,int ag,int rg,int md,int dc,int *out_t,int *out_d){
    int tot=0,dcy=0;for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+md*20000+dc*5000;
    rst<<<1,1>>>();iA<<<ag,bs>>>(da,NA,0.5f,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
    for(int step=0;step<STEPS;step++){seq<<<ag,bs>>>(da,NA,dr,NR,s,step,md,dc);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
    int h1,h2;cudaMemcpyFromSymbol(&h1,d_tot,sizeof(int));cudaMemcpyFromSymbol(&h2,d_dcy,sizeof(int));tot+=h1;dcy+=h2;}
    *out_t=tot/TRIALS;*out_d=dcy/TRIALS;}

int main(){
    printf("=== Temporal Coordination ===\nAgents:%d Resources:%d Steps:%d Trials:%d\n\n",NA,NR,STEPS,TRIALS);
    int bs=256,ag=(NA+bs-1)/bs,rg=(NR+bs-1)/bs;
    A *da;R *dr;cudaMalloc(&da,NA*sizeof(A));cudaMalloc(&dr,NR*sizeof(R));
    srand(time(NULL));

    // Hooker ratio sweep
    printf("HookerRatio Total     Decayed  PerAgent\n");
    for(float hr=0.1f;hr<=0.9f;hr+=0.1f){
        int tot=0,dcy=0;for(int t=0;t<TRIALS;t++){unsigned int s=(unsigned int)time(NULL)+t*50000+(int)(hr*1000)*1000;
        rst<<<1,1>>>();iA<<<ag,bs>>>(da,NA,hr,s);iR<<<rg,bs>>>(dr,NR,s+1);cudaDeviceSynchronize();
        for(int step=0;step<STEPS;step++){seq<<<ag,bs>>>(da,NA,dr,NR,s,step,10,100);rsp<<<rg,bs>>>(dr,NR,s+step);cudaDeviceSynchronize();}
        int h1,h2;cudaMemcpyFromSymbol(&h1,d_tot,sizeof(int));cudaMemcpyFromSymbol(&h2,d_dcy,sizeof(int));tot+=h1;dcy+=h2;}
        printf("%-11.0f%-9d%-9d%-9.2f\n",hr*100,tot/TRIALS,dcy/TRIALS,(float)(tot/TRIALS)/NA);}

    // Mode comparison
    printf("\nMode              Total     Decayed  vsSim\n");
    int sim_tot;run_sim(da,dr,bs,ag,rg,&sim_tot);
    printf("Simultaneous      %-9d%-9s1.00x\n",sim_tot,"N/A");

    int delays[]={5,10,20,40,80};
    for(int d=0;d<5;d++){int tot,dcy;run_seq(da,dr,bs,ag,rg,delays[d],100,&tot,&dcy);
    printf("Seq delay=%-7d %-9d%-9d%.2fx\n",delays[d],tot,dcy,(float)tot/sim_tot);}

    // Decay sweep
    printf("\nDecayTime  Total     Decayed  vsSim\n");
    int decays[]={30,50,100,200,500};
    for(int d=0;d<5;d++){int tot,dcy;run_seq(da,dr,bs,ag,rg,10,decays[d],&tot,&dcy);
    printf("%-10d %-9d%-9d%.2fx\n",decays[d],tot,dcy,(float)tot/sim_tot);}

    cudaFree(da);cudaFree(dr);return 0;
}
