/*
 * experiment-dcs-meta.cu — Meta-evolution of DCS cooperation protocols
 * Energy model based on proven v3 DCS (5.88x/21.87x)
 * 
 * v44: DCS baseline (reproduces v3)
 * v45: Tiered sharing (quality threshold)
 * v46: Adversarial noise (% bad advice)
 * v47: Reputation-weighted sharing
 * v48: Temporal decay
 * v49: Multi-tier (guild+public)
 * v50: Best combo
 *
 * Build: /usr/local/cuda-12.6/bin/nvcc -O2 -arch=sm_87 -lcurand -o /tmp/dcs-meta /tmp/experiment-dcs-meta.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand_kernel.h>

#define AGENTS 512
#define RESOURCES 600
#define STEPS 800
#define GRID 256
#define POOL_SZ 64
#define TYPES 3

typedef struct{
    float x,y;
    int type; // 0=generalist, 1=A, 2=B, 3=C
    float energy;
    float fitness;
    unsigned int rng;
} Agent;

typedef struct {
    float x,y,quality;
    int tick;
    float cum_benefit;
    int uses;
} Hint;

__device__ unsigned int xorshift(unsigned int s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
__device__ float frand(unsigned int*s){*s=xorshift(*s);return(*s&0xFFFFFF)/16777216.0f;}

__device__ float dist2(float ax,float ay,float bx,float by){
    float dx=ax-bx,dy=ay-by;
    // Toroidal
    if(dx>GRID/2)dx-=GRID;if(dx<-GRID/2)dx+=GRID;
    if(dy>GRID/2)dy-=GRID;if(dy<-GRID/2)dy+=GRID;
    return dx*dx+dy*dy;
}

// Shared pool in global memory
__device__ Hint g_pool[POOL_SZ];
__device__ int g_pool_n;
// Guild pools: one per specialist type
__device__ Hint g_guild[TYPES][32];
__device__ int g_guild_n[TYPES];

typedef struct {
    int id;
    int dcs;
    int tiered;       // share only above threshold
    float thresh;
    int adversarial;  // % noise (0-50)
    int reputation;   // weight by reputation
    int decay;        // temporal decay
    int multitier;    // guild+public
    char name[80];
} Cfg;

__global__ void init(Agent*a,int n,int nspec,unsigned int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    a[i].x=frand(&seed)*GRID;
    a[i].y=frand(&seed)*GRID;
    a[i].energy=1.0f;
    a[i].fitness=0;
    a[i].rng=seed+i*12345;
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

    // DCS: consult shared pool
    if(cfg.dcs){
        float hq=0;int hc=0;
        int pn=g_pool_n;
        for(int s=0;s<pn&&s<POOL_SZ;s++){
            Hint*h=&g_pool[s];
            float age=(float)(t-h->tick);
            float w=1.0f;
            if(cfg.decay) w=expf(-age*0.005f);
            if(cfg.reputation && h->uses>0)
                w*=fminf(h->cum_benefit/h->uses,1.0f);
            if(cfg.adversarial>0){
                unsigned int rs=ag->rng+s*7+i;
                if((frand(&rs)*100)<(unsigned)cfg.adversarial) continue;
            }
            float d2=dist2(ag->x,ag->y,h->x,h->y);
            if(d2<900.0f){hq+=h->quality*w;hc++;}
        }
        if(hc>0){hq/=hc;dcs_boost=hq*2.0f;}
    }

    // Guild pool
    if(cfg.multitier && ag->type>=1 && ag->type<=3){
        int gt=ag->type-1;
        int gn=g_guild_n[gt];
        for(int g=0;g<gn&&g<32;g++){
            Hint*h=&g_guild[gt][g];
            float d2=dist2(ag->x,ag->y,h->x,h->y);
            if(d2<400.0f) dcs_boost+=h->quality*0.5f;
        }
    }

    float eff_grab=grab*(1.0f+dcs_boost);

    // Find best resource
    float bd=1e9;int bj=-1;
    for(int j=0;j<nr;j++){
        float d2=dist2(ag->x,ag->y,rx[j],ry[j]);
        float eg2=eff_grab*eff_grab;
        if(d2<eg2&&d2<bd){bd=d2;bj=j;}
    }

    if(bj>=0){
        float val=rv[bj]*(1.0f+dcs_boost*0.5f);
        ag->fitness+=val;
        ag->energy=fminf(1.0f,ag->energy+0.1f);

        // Share if specialist
        if(ag->type>=1 && cfg.dcs){
            float q=fminf(val/2.0f,1.0f);
            if(cfg.tiered && q<cfg.thresh){
                // skip low quality
            } else if(g_pool_n<POOL_SZ){
                int si=atomicAdd(&g_pool_n,1);
                if(si<POOL_SZ){
                    g_pool[si].x=rx[bj];g_pool[si].y=ry[bj];
                    g_pool[si].quality=q;g_pool[si].tick=t;
                    g_pool[si].cum_benefit=0;g_pool[si].uses=0;
                }
            }
            // Guild
            if(cfg.multitier){
                int gt=ag->type-1;
                int gc=atomicAdd(&g_guild_n[gt],1);
                if(gc<32){
                    g_guild[gt][gc].x=rx[bj];g_guild[gt][gc].y=ry[bj];
                    g_guild[gt][gc].quality=q;g_guild[gt][gc].tick=t;
                }
            }
        }

        // Respawn
        rx[bj]=frand(&ag->rng)*GRID;
        ry[bj]=frand(&ag->rng)*GRID;
        rv[bj]=0.5f+frand(&ag->rng)*1.5f;
    }

    // Move toward target or random
    if(bj>=0){
        float dx=rx[bj]-ag->x,dy=ry[bj]-ag->y;
        if(dx>GRID/2)dx-=GRID;if(dx<-GRID/2)dx+=GRID;
        if(dy>GRID/2)dy-=GRID;if(dy<-GRID/2)dy+=GRID;
        float d=sqrtf(dx*dx+dy*dy);
        if(d>0.1f){ag->x+=dx/d*2.0f;ag->y+=dy/d*2.0f;}
    } else {
        ag->x+=(frand(&ag->rng)-.5f)*3.0f;
        ag->y+=(frand(&ag->rng)-.5f)*3.0f;
    }
    if(ag->x<0)ag->x+=GRID;if(ag->x>=GRID)ag->x-=GRID;
    if(ag->y<0)ag->y+=GRID;if(ag->y>=GRID)ag->y-=GRID;

    ag->energy-=0.02f;
    ag->energy*=0.9998f;
}

__global__ void reset_pools(){
    g_pool_n=0;
    for(int i=0;i<POOL_SZ;i++){
        g_pool[i].x=0;g_pool[i].y=0;g_pool[i].quality=0;
        g_pool[i].tick=0;g_pool[i].cum_benefit=0;g_pool[i].uses=0;
    }
    for(int t=0;t<TYPES;t++){
        g_guild_n[t]=0;
        for(int i=0;i<32;i++){
            g_guild[t][i].x=0;g_guild[t][i].y=0;g_guild[t][i].quality=0;
            g_guild[t][i].tick=0;g_guild[t][i].cum_benefit=0;g_guild[t][i].uses=0;
        }
    }
}

__global__ void results(Agent*a,int n,float*out,int*alive){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n){out[i]=0;alive[i]=0;return;}
    out[i]=a[i].fitness;
    alive[i]=a[i].energy>0.1f?1:0;
}

void reset_gpu_pools(){
    reset_pools<<<1,1>>>();
    cudaDeviceSynchronize();
}

void run(Cfg cfg,float*rx,float*ry,float*rv,Agent*da,float*df,int*dal){
    reset_gpu_pools();
    int nb=(AGENTS+255)/256;
    init<<<nb,256>>>(da,AGENTS,AGENTS*60/100,cfg.id*42);
    for(int t=0;t<STEPS;t++) step<<<nb,256>>>(da,AGENTS,rx,ry,rv,RESOURCES,t,cfg);
    results<<<nb,256>>>(da,AGENTS,df,dal);
}

int main(){
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DCS META-EVOLUTION v44-v50\n");
    printf("  %d agents, %d resources, %d steps\n",AGENTS,RESOURCES,STEPS);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Allocate
    Agent*da;cudaMalloc(&da,AGENTS*sizeof(Agent));
    float*rx,*ry,*rv,*df;int*dal;
    cudaMalloc(&rx,RESOURCES*sizeof(float));
    cudaMalloc(&ry,RESOURCES*sizeof(float));
    cudaMalloc(&rv,RESOURCES*sizeof(float));
    cudaMalloc(&df,AGENTS*sizeof(float));
    cudaMalloc(&dal,AGENTS*sizeof(int));

    // Init resources
    float hrx[RESOURCES],hry[RESOURCES],hrv[RESOURCES];
    srand(42);
    for(int i=0;i<RESOURCES;i++){
        hrx[i]=((float)rand()/RAND_MAX)*GRID;
        hry[i]=((float)rand()/RAND_MAX)*GRID;
        hrv[i]=0.5f+((float)rand()/RAND_MAX)*1.5f;
    }
    cudaMemcpy(rx,hrx,sizeof(hrx),cudaMemcpyHostToDevice);
    cudaMemcpy(ry,hry,sizeof(hry),cudaMemcpyHostToDevice);
    cudaMemcpy(rv,hrv,sizeof(hrv),cudaMemcpyHostToDevice);

    float hf[AGENTS];int ha[AGENTS];
    int nspec=AGENTS*60/100;

    // Define experiments
    Cfg exps[]={
        {0,0,0,0,0,0,0,0,"CONTROL: No DCS"},
        {44,1,0,0,0,0,0,0,"v44: DCS baseline"},
        // v45: tiered
        {450,1,1,0.3f,0,0,0,0,"v45a: Tiered >0.3"},
        {451,1,1,0.5f,0,0,0,0,"v45b: Tiered >0.5"},
        {452,1,1,0.7f,0,0,0,0,"v45c: Tiered >0.7"},
        // v46: adversarial
        {460,1,0,0,10,0,0,0,"v46a: 10% noise"},
        {461,1,0,0,25,0,0,0,"v46b: 25% noise"},
        {462,1,0,0,50,0,0,0,"v46c: 50% noise"},
        // v47: reputation
        {470,1,0,0,0,1,0,0,"v47a: Reputation"},
        {471,1,1,0.5f,0,1,0,0,"v47b: Rep+Tiered0.5"},
        // v48: decay
        {480,1,0,0,0,0,1,0,"v48a: Decay"},
        {481,1,1,0.5f,0,0,1,0,"v48b: Decay+Tiered"},
        // v49: multitier
        {490,1,0,0,0,0,0,1,"v49a: MultiTier"},
        {491,1,1,0.3f,0,0,0,1,"v49b: MultiTier+Tiered"},
        {492,1,0,0,0,1,0,1,"v49c: MultiTier+Rep"},
        // v50: combos
        {501,1,1,0.3f,0,1,1,0,"v50a: Tiered+Rep+Decay"},
        {502,1,1,0.3f,0,1,0,1,"v50b: Tiered+Rep+Guild"},
        {503,1,1,0.5f,10,1,0,0,"v50c: Tiered+Rep+Noise"},
        {504,1,1,0.3f,0,1,1,1,"v50d: ALL features"},
    };
    int ne=sizeof(exps)/sizeof(exps[0]);

    for(int e=0;e<ne;e++){
        run(exps[e],rx,ry,rv,da,df,dal);
        cudaMemcpy(hf,df,AGENTS*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(ha,dal,AGENTS*sizeof(int),cudaMemcpyDeviceToHost);
        float tot=0,spec=0,gen=0;int al=0,sa=0,ga=0;
        for(int i=0;i<AGENTS;i++){
            if(ha[i]){tot+=hf[i];al++;
                if(i<nspec){spec+=hf[i];sa++;}else{gen+=hf[i];ga++;}
            }
        }
        printf("  %-45s avg=%6.1f spec=%6.1f gen=%6.1f alive=%3d/%d\n",
               exps[e].name,
               al?tot/al:0, sa?spec/sa:0, ga?gen/ga:0, al, AGENTS);
    }

    cudaFree(da);cudaFree(rx);cudaFree(ry);cudaFree(rv);cudaFree(df);cudaFree(dal);
    printf("\n═══════════════════════════════════════════════════════════════\n");
    return 0;
}
