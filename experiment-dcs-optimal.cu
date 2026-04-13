/*
 * experiment-dcs-optimal.cu — DCS optimal protocol (clean rewrite)
 * Build: /usr/local/cuda-12.6/bin/nvcc -O2 -arch=sm_87 -o /tmp/dcs-optimal /tmp/experiment-dcs-optimal.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXA 2048
#define MAXR 2000
#define STEPS 800
#define GRID 256
#define POOL_SZ 256
#define TYPES 3
#define TRIALS 7

typedef struct{float x,y;int type;float energy;float fitness;unsigned int rng;} Agent;
typedef struct{float x,y,quality;int tick;} Hint;

__device__ unsigned int xs(unsigned int s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
__device__ float fr(unsigned int*s){*s=xs(*s);return(*s&0xFFFFFF)/16777216.0f;}
__device__ float dist2(float ax,float ay,float bx,float by){
    float dx=ax-bx,dy=ay-by;
    if(dx>GRID/2)dx-=GRID;if(dx<-GRID/2)dx+=GRID;
    if(dy>GRID/2)dy-=GRID;if(dy<-GRID/2)dy+=GRID;
    return dx*dx+dy*dy;
}

__device__ Hint g_pool[POOL_SZ]; __device__ int g_pool_n;
__device__ Hint g_guild[TYPES][256]; __device__ int g_guild_n[TYPES];

// Config stored in constant memory
__constant__ int C_dcs, C_tiered, C_thresh, C_multitier, C_cross_guild;
__constant__ float C_guild_r;
__constant__ int C_guild_sz, C_pub_sz, C_nspec;

__global__ void reset_pools(){
    g_pool_n=0;
    for(int i=0;i<POOL_SZ;i++){g_pool[i].x=0;g_pool[i].y=0;g_pool[i].quality=0;g_pool[i].tick=0;}
    for(int t=0;t<TYPES;t++){
        g_guild_n[t]=0;
        for(int i=0;i<256;i++){g_guild[t][i].x=0;g_guild[t][i].y=0;g_guild[t][i].quality=0;g_guild[t][i].tick=0;}
    }
}

__global__ void init(Agent*a,int n,int nspec,unsigned int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    a[i].x=fr(&seed)*GRID;a[i].y=fr(&seed)*GRID;
    a[i].energy=1.0f;a[i].fitness=0;a[i].rng=seed+i*12345;
    int spt=nspec/TYPES;
    a[i].type=(i<spt)?1:(i<spt*2)?2:(i<nspec)?3:0;
}

__global__ void step(Agent*a,int n,float*rx,float*ry,float*rv,int nr,int t){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    Agent*ag=&a[i];
    if(ag->energy<.15f){ag->energy+=.001f;return;}

    float grab=3.0f, boost=0.0f;
    unsigned int lr=ag->rng;

    // Public pool
    if(C_dcs){
        float hq=0;int hc=0;int pn=g_pool_n;
        for(int s=0;s<pn&&s<C_pub_sz;s++){
            float d2=dist2(ag->x,ag->y,g_pool[s].x,g_pool[s].y);
            if(d2<900.0f){hq+=g_pool[s].quality;hc++;}
        }
        if(hc>0) boost+=hq/hc*1.5f;
    }

    // Guild pool (specialists only)
    if(C_multitier && ag->type>=1 && ag->type<=3){
        int gt=ag->type-1;int gn=g_guild_n[gt];
        float gr2=C_guild_r*C_guild_r;
        for(int g=0;g<gn&&g<C_guild_sz;g++){
            float d2=dist2(ag->x,ag->y,g_guild[gt][g].x,g_guild[gt][g].y);
            if(d2<gr2) boost+=g_guild[gt][g].quality*0.5f;
        }
    }

    // Cross-guild: generalists read all guilds
    if(C_cross_guild && ag->type==0){
        for(int gt=0;gt<TYPES;gt++){
            int gn=g_guild_n[gt];
            for(int g=0;g<gn&&g<32;g++){
                float d2=dist2(ag->x,ag->y,g_guild[gt][g].x,g_guild[gt][g].y);
                if(d2<400.0f) boost+=g_guild[gt][g].quality*0.3f;
            }
        }
    }

    boost=fminf(boost,5.0f);
    float eg=grab*(1.0f+boost), eg2=eg*eg;
    float bd=1e9;int bj=-1;
    for(int j=0;j<nr;j++){
        float d2=dist2(ag->x,ag->y,rx[j],ry[j]);
        if(d2<eg2&&d2<bd){bd=d2;bj=j;}
    }

    if(bj>=0){
        ag->fitness+=rv[bj];
        ag->energy=fminf(1.0f,ag->energy+0.1f);

        if(ag->type>=1 && C_dcs){
            float q=fminf(rv[bj]/2.0f,1.0f);
            int share = !(C_tiered && q<C_thresh);
            if(share && g_pool_n<POOL_SZ){
                int si=atomicAdd(&g_pool_n,1);
                if(si<POOL_SZ){g_pool[si].x=rx[bj];g_pool[si].y=ry[bj];g_pool[si].quality=q;g_pool[si].tick=t;}
            }
            if(share && C_multitier){
                int gt=ag->type-1;
                int gc=atomicAdd(&g_guild_n[gt],1);
                if(gc<C_guild_sz){g_guild[gt][gc].x=rx[bj];g_guild[gt][gc].y=ry[bj];g_guild[gt][gc].quality=q;g_guild[gt][gc].tick=t;}
            }
        }
        unsigned int rs=ag->rng+bj*31;
        rx[bj]=fr(&rs)*GRID;ry[bj]=fr(&rs)*GRID;rv[bj]=0.5f+fr(&rs)*1.5f;
    }

    if(bj>=0){
        float dx=rx[bj]-ag->x,dy=ry[bj]-ag->y;
        if(dx>GRID/2)dx-=GRID;if(dx<-GRID/2)dx+=GRID;
        if(dy>GRID/2)dy-=GRID;if(dy<-GRID/2)dy+=GRID;
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0.1f){ag->x+=dx/d*2.0f;ag->y+=dy/d*2.0f;}
    } else {ag->x+=(fr(&lr)-.5f)*3.0f;ag->y+=(fr(&lr)-.5f)*3.0f;}
    if(ag->x<0)ag->x+=GRID;if(ag->x>=GRID)ag->x-=GRID;
    if(ag->y<0)ag->y+=GRID;if(ag->y>=GRID)ag->y-=GRID;
    ag->energy-=0.02f;ag->energy*=0.9998f;
}

__global__ void results(Agent*a,int n,float*out,int*alive){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n){out[i]=0;alive[i]=0;return;}
    out[i]=a[i].fitness;alive[i]=a[i].energy>0.1f?1:0;
}

void set_cfg(int dcs,int tiered,int thresh,int multitier,int cross,
             float gr,int gsz,int psz,int nspec){
    cudaMemcpyToSymbol(C_dcs,&dcs,sizeof(int));
    cudaMemcpyToSymbol(C_tiered,&tiered,sizeof(int));
    cudaMemcpyToSymbol(C_thresh,&thresh,sizeof(int));
    cudaMemcpyToSymbol(C_multitier,&multitier,sizeof(int));
    cudaMemcpyToSymbol(C_cross_guild,&cross,sizeof(int));
    cudaMemcpyToSymbol(C_guild_r,&gr,sizeof(float));
    cudaMemcpyToSymbol(C_guild_sz,&gsz,sizeof(int));
    cudaMemcpyToSymbol(C_pub_sz,&psz,sizeof(int));
    cudaMemcpyToSymbol(C_nspec,&nspec,sizeof(int));
}

void run(int id,int N,int NS,int NR,float*rx,float*ry,float*rv,
         Agent*da,float*df,int*dal,float*oa,float*os,float*og,int*oal){
    float ta=0,ts=0,tg=0;int tal=0;
    for(int tr=0;tr<TRIALS;tr++){
        reset_pools<<<1,1>>>();cudaDeviceSynchronize();
        int nb=(N+255)/256;
        init<<<nb,256>>>(da,N,NS,id*100000+tr*9999);
        for(int t=0;t<STEPS;t++) step<<<nb,256>>>(da,N,rx,ry,rv,NR,t);
        results<<<nb,256>>>(da,N,df,dal);
        float hf[MAXA];int ha[MAXA];
        cudaMemcpy(hf,df,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(ha,dal,N*sizeof(int),cudaMemcpyDeviceToHost);
        float tot=0,sp=0,gn=0;int al=0,sa=0,ga=0;
        for(int i=0;i<N;i++){if(ha[i]){tot+=hf[i];al++;if(i<NS){sp+=hf[i];sa++;}else{gn+=hf[i];ga++;}}}
        ta+=tot;ts+=sp;tg+=gn;tal+=al;
    }
    *oa=ta/TRIALS;*os=ts/TRIALS;*og=tg/TRIALS;*oal=tal/TRIALS;
}

void pr(const char*name,int id,int dcs,int tiered,int thresh,int multitier,int cross,
        float gr,int gsz,int psz,int NS,int N,float*rx,float*ry,float*rv,
        Agent*da,float*df,int*dal){
    set_cfg(dcs,tiered,thresh,multitier,cross,gr,gsz,psz,NS);
    float av,sp,gn;int al;
    run(id,N,NS,600,rx,ry,rv,da,df,dal,&av,&sp,&gn,&al);
    printf("  %-50s avg=%8.0f spec=%8.0f gen=%7.0f\n",name,av,sp,gn);
}

int main(){
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  DCS OPTIMAL — Clean Rewrite, %d trials × %d steps\n",TRIALS,STEPS);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int N=512,NR=600;
    Agent*da;cudaMalloc(&da,MAXA*sizeof(Agent));
    float*rx,*ry,*rv,*df;int*dal;
    cudaMalloc(&rx,MAXR*sizeof(float));cudaMalloc(&ry,MAXR*sizeof(float));
    cudaMalloc(&rv,MAXR*sizeof(float));cudaMalloc(&df,MAXA*sizeof(float));
    cudaMalloc(&dal,MAXA*sizeof(int));

    float hrx[MAXR],hry[MAXR],hrv[MAXR];
    srand(42);
    for(int i=0;i<MAXR;i++){hrx[i]=((float)rand()/RAND_MAX)*GRID;hry[i]=((float)rand()/RAND_MAX)*GRID;hrv[i]=0.5f+((float)rand()/RAND_MAX)*1.5f;}
    cudaMemcpy(rx,hrx,MAXR*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(ry,hry,MAXR*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(rv,hrv,MAXR*sizeof(float),cudaMemcpyHostToDevice);

    // Baselines
    printf("── BASELINES ──\n");
    pr("CONTROL",0, 0,0,0,0,0, 50,0,0, N*60/100,N, rx,ry,rv,da,df,dal);
    pr("DCS public only",44, 1,0,0,0,0, 50,0,64, N*60/100,N, rx,ry,rv,da,df,dal);
    pr("Guild only (128,r50)",542, 1,0,0,1,0, 50,128,0, N*60/100,N, rx,ry,rv,da,df,dal);
    pr("Guild+Tiered20+Pub16 (v3best)",543, 1,1,20,1,0, 50,128,16, N*30/100,N, rx,ry,rv,da,df,dal);
    printf("\n");

    // v64: Specialist ratio
    printf("── v64: Specialist Ratio ──\n");
    int pcts[]={5,10,15,20,25,30,40,50,60,70,80,90,95};
    for(int i=0;i<13;i++){
        char buf[80];snprintf(buf,80,"Spec=%d%%",pcts[i]);
        pr(buf,640+i, 1,1,20,1,0, 50,128,16, N*pcts[i]/100,N, rx,ry,rv,da,df,dal);
    }
    printf("\n");

    // v65: Cross-guild
    printf("── v65: Cross-Guild & Exclusive ──\n");
    int ns30=N*30/100;
    pr("Baseline (no cross)",650, 1,1,20,1,0, 50,128,16, ns30,N, rx,ry,rv,da,df,dal);
    pr("Cross-guild ON",651, 1,1,20,1,1, 50,128,16, ns30,N, rx,ry,rv,da,df,dal);
    pr("Guild exclusive (gen blocked)",652, 1,1,20,1,0, 50,128,0, ns30,N, rx,ry,rv,da,df,dal);
    pr("Cross+Exclusive",653, 1,1,20,1,1, 50,128,0, ns30,N, rx,ry,rv,da,df,dal);
    printf("\n");

    // v66: Tiered threshold
    printf("── v66: Tiered Threshold ──\n");
    int thrs[]={0,10,20,30,50,70};
    for(int i=0;i<6;i++){
        char buf[80];snprintf(buf,80,"Tiered>=%d%%",thrs[i]);
        pr(buf,660+i, 1,thrs[i]>0?1:0,thrs[i],1,0, 50,128,16, ns30,N, rx,ry,rv,da,df,dal);
    }
    printf("\n");

    // v67: Public pool size
    printf("── v67: Public Pool Size ──\n");
    int pszs[]={0,8,16,32,64,128};
    for(int i=0;i<6;i++){
        char buf[80];snprintf(buf,80,"Pub=%d",pszs[i]);
        int dcs_flag = pszs[i]==0 ? 0 : 1;
        pr(buf,670+i, dcs_flag,1,20,1,0, 50,128,pszs[i], ns30,N, rx,ry,rv,da,df,dal);
    }
    printf("\n");

    // v68: Grid sweep
    printf("── v68: Guild Size × Radius Grid ──\n");
    printf("  %-12s","");
    int gsizes[]={16,32,64,128};
    for(int j=0;j<4;j++) printf("%12d",gsizes[j]);
    printf("\n");
    float gradii[]={20,30,40,50};
    for(int i=0;i<4;i++){
        printf("  radius=%-4d ",(int)gradii[i]);
        for(int j=0;j<4;j++){
            set_cfg(1,1,20,1,0,gradii[i],gsizes[j],16,ns30);
            float av,sp,gn;int al;
            run(680+i*4+j,N,ns30,600,rx,ry,rv,da,df,dal,&av,&sp,&gn,&al);
            printf("%12.0f",av);
        }
        printf("\n");
    }
    printf("\n");

    // v69: Final top 5
    printf("── v69: FINAL Top Candidates ──\n");
    pr("BEST: G128r50+T20+P16+S30",690, 1,1,20,1,0, 50,128,16, ns30,N, rx,ry,rv,da,df,dal);
    pr("BEST+CrossGuild",691, 1,1,20,1,1, 50,128,16, ns30,N, rx,ry,rv,da,df,dal);
    pr("G128r50+T10+P16+S20",692, 1,1,10,1,0, 50,128,16, N*20/100,N, rx,ry,rv,da,df,dal);
    pr("G64r40+T20+P16+S30",693, 1,1,20,1,0, 40,64,16, ns30,N, rx,ry,rv,da,df,dal);
    pr("G128r50+T20+P0+S30 (guild only)",694, 1,1,20,1,0, 50,128,0, ns30,N, rx,ry,rv,da,df,dal);

    cudaFree(da);cudaFree(rx);cudaFree(ry);cudaFree(rv);cudaFree(df);cudaFree(dal);
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    return 0;
}
