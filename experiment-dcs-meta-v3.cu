/*
 * experiment-dcs-meta-v3.cu — DCS protocol refinement
 * 
 * Key findings from v2:
 * - Guild pool size 128 >> 32 (bigger better, no saturation)
 * - Guild radius 50 >> 20 (wider search wins)
 * - 20% specialists > 60% (fewer specialists, more generalists benefit)
 * - Guild+Public (no Rep) > Guild+Rep alone (reputation needs filtering)
 * - Guild+Rep+Tiered0.5 = best combo
 * - Protocol creates positive feedback loop at high resource density
 *
 * v58: Specialist ratio fine sweep (5%-50%)
 * v59: Tiered threshold sweep with Guild+Rep
 * v60: Public pool size sweep (is public needed at all?)
 * v61: Agent count × resource density interaction
 * v62: Fitness capping (prevent compounding) — stability test
 * v63: Optimal protocol — parameterized search
 *
 * Build: /usr/local/cuda-12.6/bin/nvcc -O2 -arch=sm_87 -o /tmp/dcs-meta-v3 /tmp/experiment-dcs-meta-v3.cu
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
#define TRIALS 5

typedef struct{float x,y;int type;float energy;float fitness;unsigned int rng;} Agent;
typedef struct{float x,y,quality;int tick;float cum_benefit;int uses;} Hint;

__device__ unsigned int xs(unsigned int s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
__device__ float fr(unsigned int*s){*s=xs(*s);return(*s&0xFFFFFF)/16777216.0f;}
__device__ float dist2(float ax,float ay,float bx,float by){
    float dx=ax-bx,dy=ay-by;
    if(dx>GRID/2)dx-=GRID;if(dx<-GRID/2)dx+=GRID;
    if(dy>GRID/2)dy-=GRID;if(dy<-GRID/2)dy+=GRID;
    return dx*dx+dy*dy;
}

__device__ Hint g_pool[POOL_SZ];
__device__ int g_pool_n;
__device__ Hint g_guild[TYPES][256];
__device__ int g_guild_n[TYPES];

typedef struct {
    int id, dcs, tiered, adversarial, reputation, decay, multitier;
    int thresh_x100;
    float guild_radius;
    int guild_size;
    int public_size;
    float fitness_cap; // 0 = no cap
    int nspec, nres, n;
    char name[80];
} Cfg;

__global__ void reset_pools(){
    g_pool_n=0;
    for(int i=0;i<POOL_SZ;i++){Hint*h=&g_pool[i];h->x=0;h->y=0;h->quality=0;
        h->tick=0;h->cum_benefit=0;h->uses=0;}
    for(int t=0;t<TYPES;t++){
        g_guild_n[t]=0;
        for(int i=0;i<256;i++){Hint*h=&g_guild[t][i];h->x=0;h->y=0;h->quality=0;
            h->tick=0;h->cum_benefit=0;h->uses=0;}
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

__global__ void step(Agent*a,int n,float*rx,float*ry,float*rv,int nr,
                     int t,Cfg cfg){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    Agent*ag=&a[i];
    if(ag->energy<.15f){ag->energy+=.001f;return;}

    float grab=3.0f;
    float dcs_boost=0.0f;
    unsigned int lrng=ag->rng;

    // Public pool consultation
    if(cfg.dcs){
        float hq=0;int hc=0;
        int pn=g_pool_n;
        for(int s=0;s<pn&&s<cfg.public_size;s++){
            Hint*h=&g_pool[s];
            float age=(float)(t-h->tick);
            float w=1.0f;
            if(cfg.decay) w=expf(-age*0.005f);
            if(cfg.reputation && h->uses>0)
                w*=fminf(h->cum_benefit/(float)h->uses,1.0f);
            if(cfg.adversarial>0)
                if((fr(&lrng)*100)<(unsigned)cfg.adversarial) continue;
            float d2=dist2(ag->x,ag->y,h->x,h->y);
            if(d2<900.0f){hq+=h->quality*w;hc++;}
        }
        if(hc>0){hq/=hc;dcs_boost=hq*2.0f;}
    }

    // Guild pool consultation
    if(cfg.multitier && ag->type>=1 && ag->type<=3){
        int gt=ag->type-1;
        int gn=g_guild_n[gt];
        float gr2=cfg.guild_radius*cfg.guild_radius;
        for(int g=0;g<gn&&g<cfg.guild_size;g++){
            Hint*h=&g_guild[gt][g];
            float d2=dist2(ag->x,ag->y,h->x,h->y);
            if(d2<gr2){
                float w=1.0f;
                if(cfg.reputation && h->uses>0)
                    w*=fminf(h->cum_benefit/(float)h->uses,1.0f);
                dcs_boost+=h->quality*w*0.5f;
            }
        }
    }

    // Cap boost to prevent runaway
    dcs_boost=fminf(dcs_boost,5.0f);
    float eff_grab=grab*(1.0f+dcs_boost);

    float bd=1e9;int bj=-1;
    float eg2=eff_grab*eff_grab;
    for(int j=0;j<nr;j++){
        float d2=dist2(ag->x,ag->y,rx[j],ry[j]);
        if(d2<eg2&&d2<bd){bd=d2;bj=j;}
    }

    if(bj>=0){
        float val=rv[bj]; // base value, no DCS multiplier on collection
        if(cfg.fitness_cap>0) val=fminf(val,cfg.fitness_cap);
        ag->fitness+=val;
        ag->energy=fminf(1.0f,ag->energy+0.1f);

        // Share (specialists only)
        if(ag->type>=1 && cfg.dcs){
            float q=fminf(val/2.0f,1.0f);
            if(cfg.tiered && q<cfg.thresh_x100/100.0f){}
            else if(g_pool_n<POOL_SZ){
                int si=atomicAdd(&g_pool_n,1);
                if(si<POOL_SZ){
                    g_pool[si].x=rx[bj];g_pool[si].y=ry[bj];
                    g_pool[si].quality=q;g_pool[si].tick=t;
                }
            }
            if(cfg.multitier){
                int gt=ag->type-1;
                int gc=atomicAdd(&g_guild_n[gt],1);
                if(gc<cfg.guild_size){
                    g_guild[gt][gc].x=rx[bj];g_guild[gt][gc].y=ry[bj];
                    g_guild[gt][gc].quality=q;g_guild[gt][gc].tick=t;
                }
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
    } else {
        ag->x+=(fr(&lrng)-.5f)*3.0f;ag->y+=(fr(&lrng)-.5f)*3.0f;
    }
    if(ag->x<0)ag->x+=GRID;if(ag->x>=GRID)ag->x-=GRID;
    if(ag->y<0)ag->y+=GRID;if(ag->y>=GRID)ag->y-=GRID;
    ag->energy-=0.02f;ag->energy*=0.9998f;
}

__global__ void results(Agent*a,int n,float*out,int*alive){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n){out[i]=0;alive[i]=0;return;}
    out[i]=a[i].fitness;
    alive[i]=a[i].energy>0.1f?1:0;
}

void run(Cfg cfg,float*rx,float*ry,float*rv,Agent*da,float*df,int*dal,
         float*out_avg,float*out_spec,float*out_gen,int*out_alive){
    float ta=0,ts=0,tg=0;int tal=0;
    for(int trial=0;trial<TRIALS;trial++){
        reset_pools<<<1,1>>>();cudaDeviceSynchronize();
        int nb=(cfg.n+255)/256;
        init<<<nb,256>>>(da,cfg.n,cfg.nspec,cfg.id*100000+trial*9999);
        for(int t=0;t<STEPS;t++) step<<<nb,256>>>(da,cfg.n,rx,ry,rv,cfg.nres,t,cfg);
        results<<<nb,256>>>(da,cfg.n,df,dal);
        float hf[MAXA];int ha[MAXA];
        cudaMemcpy(hf,df,cfg.n*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(ha,dal,cfg.n*sizeof(int),cudaMemcpyDeviceToHost);
        float tot=0,spec=0,gen=0;int al=0,sa=0,ga=0;
        for(int i=0;i<cfg.n;i++){
            if(ha[i]){tot+=hf[i];al++;
                if(i<cfg.nspec){spec+=hf[i];sa++;}else{gen+=hf[i];ga++;}}
        }
        ta+=tot;ts+=spec;tg+=gen;tal+=al;
    }
    *out_avg=ta/TRIALS;*out_spec=ts/TRIALS;*out_gen=tg/TRIALS;*out_alive=tal/TRIALS;
}

void pr(Cfg cfg,float*rx,float*ry,float*rv,Agent*da,float*df,int*dal){
    float av,sp,gn;int al;
    run(cfg,rx,ry,rv,da,df,dal,&av,&sp,&gn,&al);
    printf("  %-48s avg=%8.1f spec=%8.1f gen=%7.1f al=%d\n",cfg.name,av,sp,gn,al);
}

int main(){
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  DCS META-EVOLUTION v3 — Refinement\n");
    printf("  %d trials × %d steps. Fitness capped (no DCS value mult).\n",TRIALS,STEPS);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    int N=512,NS=307,NR=600; // default 60% spec

    Agent*da;cudaMalloc(&da,MAXA*sizeof(Agent));
    float*rx,*ry,*rv,*df;int*dal;
    cudaMalloc(&rx,MAXR*sizeof(float));cudaMalloc(&ry,MAXR*sizeof(float));
    cudaMalloc(&rv,MAXR*sizeof(float));cudaMalloc(&df,MAXA*sizeof(float));
    cudaMalloc(&dal,MAXA*sizeof(int));

    float hrx[MAXR],hry[MAXR],hrv[MAXR];
    srand(42);
    for(int i=0;i<MAXR;i++){
        hrx[i]=((float)rand()/RAND_MAX)*GRID;
        hry[i]=((float)rand()/RAND_MAX)*GRID;
        hrv[i]=0.5f+((float)rand()/RAND_MAX)*1.5f;
    }
    cudaMemcpy(rx,hrx,MAXR*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(ry,hry,MAXR*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(rv,hrv,MAXR*sizeof(float),cudaMemcpyHostToDevice);

    // Baseline configs
    Cfg ctrl={0,0,0,0,0,0,0,0,20,32,64,0,NS,NR,N,"CONTROL: No DCS"};
    Cfg base={44,1,0,0,0,0,0,0,20,32,64,0,NS,NR,N,"DCS baseline (public only)"};
    Cfg guild={542,1,0,0,0,0,1,0,20,128,64,0,NS,NR,N,"Guild only"};
    Cfg optimal={544,1,0,0,0,0,1,0,50,128,128,0,NS,NR,N,"Guild+Rep+Tiered0.5 (v2 best)"};

    printf("── BASELINES ──\n");
    pr(ctrl,rx,ry,rv,da,df,dal);
    pr(base,rx,ry,rv,da,df,dal);
    pr(guild,rx,ry,rv,da,df,dal);
    pr(optimal,rx,ry,rv,da,df,dal);
    printf("\n");

    // ─── v58: Specialist ratio fine sweep ───
    printf("── v58: Specialist Ratio (Guild+Rep+Tiered0.5) ──\n");
    int pcts[]={5,10,15,20,25,30,40,50,60,70,80,90,95};
    for(int i=0;i<13;i++){
        int ns=N*pcts[i]/100;
        Cfg c={580+i,1,0,0,0,0,1,50,50,128,128,0,ns,NR,N,"v58"};
        snprintf(c.name,80,"Spec=%d%%",pcts[i]);
        pr(c,rx,ry,rv,da,df,dal);
    }
    printf("\n");

    // ─── v59: Tiered threshold sweep ───
    printf("── v59: Tiered Threshold (Guild+Rep) ──\n");
    int threshes[]={0,10,20,30,40,50,60,70,80,90};
    for(int i=0;i<10;i++){
        Cfg c={590+i,1,threshes[i]>0?1:0,0,0,0,1,threshes[i],50,128,128,0,NS,NR,N,"v59"};
        snprintf(c.name,80,"Tiered>=%d%%",threshes[i]);
        pr(c,rx,ry,rv,da,df,dal);
    }
    printf("\n");

    // ─── v60: Public pool size ───
    printf("── v60: Public Pool Size (Guild+Rep+Tiered0.5) ──\n");
    int psizes[]={0,16,32,64,128,256};
    for(int i=0;i<6;i++){
        Cfg c={600+i,1,1,0,0,0,1,50,50,128,psizes[i],0,NS,NR,N,"v60"};
        if(psizes[i]==0) c.dcs=0; // no public pool
        snprintf(c.name,80,"Public pool=%d",psizes[i]);
        pr(c,rx,ry,rv,da,df,dal);
    }
    printf("\n");

    // ─── v61: Resource density × agent count ───
    printf("── v61: Resource Density (per agent) with optimal protocol ──\n");
    float densities[]={0.5f,0.75f,1.0f,1.25f,1.5f,2.0f,3.0f,4.0f};
    for(int i=0;i<8;i++){
        int nr=(int)(N*densities[i]);
        if(nr>MAXR) nr=MAXR;
        Cfg c={610+i,1,1,0,0,0,1,50,50,128,128,0,NS,nr,N,"v61"};
        snprintf(c.name,80,"Res/agent=%.1f",densities[i]);
        pr(c,rx,ry,rv,da,df,dal);
    }
    printf("\n");

    // ─── v62: Does reputation actually help? ───
    printf("── v62: Reputation Isolation (Guild only, vary features) ──\n");
    Cfg iso[]={
        {620,1,0,0,0,0,1,0,50,128,128,0,NS,NR,N,"Guild only"},
        {621,1,0,0,1,0,1,0,50,128,128,0,NS,NR,N,"Guild+Rep"},
        {622,1,0,0,0,0,1,0,50,50,32,0,NS,NR,N,"Guild small (r=50,p=32)"},
        {623,1,0,0,1,0,1,0,50,50,32,0,NS,NR,N,"Guild small+Rep"},
        {624,1,0,0,0,0,1,0,20,20,8,0,NS,NR,N,"Guild tiny (r=20,p=8)"},
        {625,1,0,0,1,0,1,0,20,20,8,0,NS,NR,N,"Guild tiny+Rep"},
        {626,0,0,0,0,0,0,0,0,0,0,0,NS,NR,N,"No protocol (control)"},
        {627,1,0,0,0,0,0,0,0,0,64,0,NS,NR,N,"Public pool only (no guild)"},
        {628,1,0,0,1,0,0,0,0,0,64,0,NS,NR,N,"Public+Rep (no guild)"},
        {629,1,1,0,0,0,1,50,50,128,128,0,NS,NR,N,"Guild+Tiered0.5 (no Rep)"},
    };
    for(int i=0;i<10;i++) pr(iso[i],rx,ry,rv,da,df,dal);
    printf("\n");

    // ─── v63: The optimal parameter sweep ───
    printf("── v63: Optimal Protocol Grid (guild_size × guild_radius) ──\n");
    printf("  %-12s","");
    int gsizes[]={16,32,64,128};
    for(int j=0;j<4;j++) printf("%12d",gsizes[j]);
    printf("\n");
    int gradii[]={20,30,40,50};
    for(int i=0;i<4;i++){
        printf("  radius=%-4d ",gradii[i]);
        for(int j=0;j<4;j++){
            Cfg c={630+i*4+j,1,1,0,0,0,1,50,(float)gradii[i],gsizes[j],64,0,NS,NR,N,""};
            float av,sp,gn;int al;
            run(c,rx,ry,rv,da,df,dal,&av,&sp,&gn,&al);
            printf("%12.0f",av);
        }
        printf("\n");
    }
    printf("\n");

    cudaFree(da);cudaFree(rx);cudaFree(ry);cudaFree(rv);cudaFree(df);cudaFree(dal);
    printf("═══════════════════════════════════════════════════════════════════\n");
    return 0;
}
