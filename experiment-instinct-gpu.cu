/* experiment-instinct-no-energy.cu — Instinct-driven behavior WITHOUT energy costs
   Based on SuperInstance/flux-instinct priority reflex engine.
   8 instincts: survive, flee, guard, hoard, cooperate, teach, curious, mourn.
   Key question: our energy experiments showed biological constraints hurt (-27%).
   But what if instincts drive behavior WITHOUT energy cost? Pure priority-based decision making.
   
   Compare: instinct-driven (no energy) vs energy-driven vs baseline specialists. */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NA 1024
#define NR 128
#define MAXT 500
#define N_ARCH 4
#define N_INSTINCTS 8

__device__ __host__ unsigned int lcg(unsigned int*s){*s=*s*1103515245u+12345u;return(*s>>16)&0x7fff;}
__device__ __host__ float lcgf(unsigned int*s){return(float)lcg(s)/32768.0f;}

typedef struct{float x,y,vx,vy,energy,role[4],fitness;int arch,res_held;float tip_x,tip_y,tip_val;unsigned int rng;}Agent;
typedef struct{float x,y,value;int collected;}Resource;

// Instinct enum matching flux-instinct
#define INST_SURVIVE 0
#define INST_FLEE    1
#define INST_GUARD   2
#define INST_HOARD   3
#define INST_COOP    4
#define INST_TEACH   5
#define INST_CURIOUS 6
#define INST_MOURN   7

__global__ void init_a(Agent*a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;a[i].rng=(unsigned int)(i*2654435761u+17);a[i].x=lcgf(&a[i].rng);a[i].y=lcgf(&a[i].rng);a[i].vx=a[i].vy=0;a[i].energy=.5f+lcgf(&a[i].rng)*.5f;a[i].arch=i%N_ARCH;a[i].fitness=a[i].res_held=0;a[i].tip_x=a[i].tip_y=a[i].tip_val=0;for(int r=0;r<4;r++){float b=(r==a[i].arch)?.7f:.1f;a[i].role[r]=b+(lcgf(&a[i].rng)-.5f)*.4f;}}
__global__ void init_c(Agent*a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;a[i].rng=(unsigned int)(i*2654435761u+99917);a[i].x=lcgf(&a[i].rng);a[i].y=lcgf(&a[i].rng);a[i].vx=a[i].vy=0;a[i].energy=.5f+lcgf(&a[i].rng)*.5f;a[i].arch=i%N_ARCH;a[i].fitness=a[i].res_held=0;a[i].tip_x=a[i].tip_y=a[i].tip_val=0;for(int r=0;r<4;r++)a[i].role[r]=.25f;}
__global__ void init_r(Resource*r,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;unsigned int s=(unsigned int)(i*2654435761u+99999);r[i].x=lcgf(&s);r[i].y=lcgf(&s);r[i].value=.5f+lcgf(&s)*.5f;r[i].collected=0;}

// Standard specialist tick (baseline from v40)
__global__ void tick_s(Agent*a,Resource*r,int na,int nr,int t,int pt){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    Agent*ag=&a[i];float ep=ag->role[0],cp=ag->role[1],cm=ag->role[2],df=ag->role[3];
    float det=.03f+ep*.04f,grab=.02f+cp*.02f;float bd=det;int br=-1;
    for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}
    if(br>=0&&bd<grab){r[br].collected=1;ag->res_held++;float tb=1;for(int k=0;k<16;k++){int j=lcg(&ag->rng)%na;if(j==i||a[j].arch!=ag->arch)continue;float dx=a[j].x-ag->x,dy=a[j].y-ag->y;if(sqrtf(dx*dx+dy*dy)<.05f)tb+=a[j].role[3]*.2f;}float bn=(1+cp*.5f)*tb;ag->energy=fminf(1,ag->energy+r[br].value*.1f*bn);ag->fitness+=r[br].value*bn;}
    else if(br>=0){float dx=r[br].x-ag->x,dy=r[br].y-ag->y,d=sqrtf(dx*dx+dy*dy),sp=.008f+cp*.008f+ep*.006f;ag->vx=ag->vx*.8f+(dx/d)*sp;ag->vy=ag->vy*.8f+(dy/d)*sp;}
    else{ag->vx=ag->vx*.95f+(lcgf(&ag->rng)-.5f)*.006f*(1+ep);ag->vy=ag->vy*.95f+(lcgf(&ag->rng)-.5f)*.006f*(1+ep);}
    ag->x=fmodf(ag->x+ag->vx+1,1);ag->y=fmodf(ag->y+ag->vy+1,1);
    for(int k=0;k<32;k++){int j=lcg(&ag->rng)%na;if(j==i)continue;float dx=a[j].x-ag->x,dy=a[j].y-ag->y,dist=sqrtf(dx*dx+dy*dy);if(dist>=.06f)continue;
        if(a[j].role[2]>.5f&&cm>.2f){float jbd=.1f;int jbr=-1;for(int m=0;m<nr;m++){if(r[m].collected)continue;float mdx=r[m].x-a[j].x,mdy=r[m].y-a[j].y,md=sqrtf(mdx*mdx+mdy*mdy);if(md<jbd){jbd=md;jbr=m;}}if(jbr>=0){ag->tip_x=r[jbr].x;ag->tip_y=r[jbr].y;ag->tip_val=a[j].role[2];}}
        float infl=(a[j].arch==ag->arch)?.02f:.002f;for(int r=0;r<4;r++)ag->role[r]+=(a[j].role[r]-ag->role[r])*infl;
        float sim=0;for(int r=0;r<4;r++)sim+=1-fminf(1,fabsf(ag->role[r]-a[j].role[r]));sim/=4;if(sim>.9f){int dr=(ag->arch+1+lcg(&ag->rng)%3)%4;ag->role[dr]+=(lcgf(&ag->rng)-.5f)*.01f;}
        if(dist<.02f){ag->vx-=dx*.01f;ag->vy-=dy*.01f;}}
    int dom=0;float dv=ag->role[0];for(int r=1;r<4;r++)if(ag->role[r]>dv){dv=ag->role[r];dom=r;}if(dom==ag->arch)ag->energy=fminf(1,ag->energy+.0005f);else ag->energy*=.9995f;ag->energy*=.999f;for(int r=0;r<4;r++){if(ag->role[r]<0)ag->role[r]=0;if(ag->role[r]>1)ag->role[r]=1;}
    if(t==pt){ag->energy*=(1-.5f*(1-df*.5f));ag->x=lcgf(&ag->rng);ag->y=lcgf(&ag->rng);ag->vx=ag->vy=0;ag->tip_val=0;}
}

// Instinct-driven tick — NO energy costs, priority-based decision making
__global__ void tick_instinct(Agent*a,Resource*r,int na,int nr,int t,int pt){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    Agent*ag=&a[i];
    
    // Calculate instinct urgencies (matching flux-instinct thresholds)
    float urgency[N_INSTINCTS];
    urgency[INST_SURVIVE]=(ag->energy<.15f)?1-ag->energy/.15f:0;  // low energy
    urgency[INST_FLEE]=0;  // no predators in this sim
    urgency[INST_GUARD]=(ag->role[3]>.5f)?ag->role[3]*.5f:0;      // defensive agents guard
    urgency[INST_HOARD]=(ag->role[1]>.5f)?ag->role[1]*.6f:0;     // collector instinct
    urgency[INST_COOP]=(ag->role[2]>.5f)?ag->role[2]*.4f:0;      // communicator instinct
    urgency[INST_TEACH]=(ag->role[3]>.7f)?.2f:0;                  // teacher instinct
    urgency[INST_CURIOUS]=.15f;                                    // always present
    urgency[INST_MOURN]=0;                                         // no death events
    
    // Find highest priority instinct
    int best_inst=INST_CURIOUS;float best_urg=urgency[INST_CURIOUS];
    for(int inst=0;inst<N_INSTINCTS;inst++)if(urgency[inst]>best_urg){best_urg=urgency[inst];best_inst=inst;}
    
    // Execute instinct-driven behavior
    float det=.03f+ag->role[0]*.04f,grab=.02f+ag->role[1]*.02f;
    
    switch(best_inst){
    case INST_SURVIVE: {
        // Emergency: find nearest resource regardless
        float bd=det*2;int br=-1;
        for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}
        if(br>=0&&bd<grab*2){r[br].collected=1;ag->res_held++;ag->energy=fminf(1,ag->energy+r[br].value*.15f);ag->fitness+=r[br].value*1.5f;}
        else{float bd2=det*3;for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd2){ag->vx+=(dx/d)*.02f;ag->vy+=(dy/d)*.02f;}}}
        break;}
    case INST_HOARD: {
        // Collect aggressively, ignore tips
        float bd=det;int br=-1;
        for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}
        if(br>=0&&bd<grab){r[br].collected=1;ag->res_held++;float tb=1;for(int k=0;k<16;k++){int j=lcg(&ag->rng)%na;if(j==i||a[j].arch!=ag->arch)continue;float dx=a[j].x-ag->x,dy=a[j].y-ag->y;if(sqrtf(dx*dx+dy*dy)<.05f)tb+=a[j].role[3]*.2f;}ag->energy=fminf(1,ag->energy+r[br].value*.1f*tb);ag->fitness+=r[br].value*tb;}
        else if(br>=0){float dx=r[br].x-ag->x,dy=r[br].y-ag->y,d=sqrtf(dx*dx+dy*dy),sp=.012f;ag->vx=ag->vx*.7f+(dx/d)*sp;ag->vy=ag->vy*.7f+(dy/d)*sp;}
        break;}
    case INST_COOP: {
        // Share tips actively, then collect
        float bd=det;int br=-1;
        for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}
        // Tell nearby same-arch agents about find
        for(int k=0;k<32;k++){int j=lcg(&ag->rng)%na;if(j==i)continue;float dx=a[j].x-ag->x,dy=a[j].y-ag->y;if(sqrtf(dx*dx+dy*dy)<.06f&&a[j].arch==ag->arch){a[j].tip_x=r[br>=0?br:0].x;a[j].tip_y=r[br>=0?br:0].y;a[j].tip_val=ag->role[2];}}
        if(br>=0&&bd<grab){r[br].collected=1;ag->res_held++;ag->energy=fminf(1,ag->energy+r[br].value*.1f);ag->fitness+=r[br].value;}
        else if(br>=0){float dx=r[br].x-ag->x,dy=r[br].y-ag->y,d=sqrtf(dx*dx+dy*dy);ag->vx=ag->vx*.8f+(dx/d)*.008f;ag->vy=ag->vy*.8f+(dy/d)*.008f;}
        break;}
    case INST_GUARD: {
        // Stay near cluster, collect slowly
        float bd=det*.7f;int br=-1;
        for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}
        if(br>=0&&bd<grab){r[br].collected=1;ag->res_held++;ag->fitness+=r[br].value;}
        // Attract to same-arch neighbors
        for(int k=0;k<16;k++){int j=lcg(&ag->rng)%na;if(j==i||a[j].arch!=ag->arch)continue;float dx=a[j].x-ag->x,dy=a[j].y-ag->y;if(sqrtf(dx*dx+dy*dy)<.1f){ag->vx+=dx*.003f;ag->vy+=dy*.003f;}}
        break;}
    case INST_TEACH: {
        // Broadcast knowledge, then collect
        float bd=det;int br=-1;
        for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}
        for(int k=0;k<32;k++){int j=lcg(&ag->rng)%na;if(j==i)continue;float dx=a[j].x-ag->x,dy=a[j].y-ag->y;if(sqrtf(dx*dx+dy*dy)<.06f){for(int r=0;r<4;r++)a[j].role[r]+=(ag->role[r]-a[j].role[r])*.05f;}} // teach roles
        if(br>=0&&bd<grab){r[br].collected=1;ag->res_held++;ag->energy=fminf(1,ag->energy+r[br].value*.1f);ag->fitness+=r[br].value;}
        break;}
    default: { // CURIOUS — wander and collect
        float bd=det;int br=-1;
        for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}
        if(br>=0&&bd<grab){r[br].collected=1;ag->res_held++;ag->energy=fminf(1,ag->energy+r[br].value*.1f);ag->fitness+=r[br].value;}
        else{ag->vx=ag->vx*.9f+(lcgf(&ag->rng)-.5f)*.015f;ag->vy=ag->vy*.9f+(lcgf(&ag->rng)-.5f)*.015f;}
        break;}
    }
    
    ag->x=fmodf(ag->x+ag->vx+1,1);ag->y=fmodf(ag->y+ag->vy+1,1);
    // Anti-convergence
    for(int k=0;k<8;k++){int j=lcg(&ag->rng)%na;if(j==i)continue;float sim=0;for(int r=0;r<4;r++)sim+=1-fminf(1,fabsf(ag->role[r]-a[j].role[r]));sim/=4;if(sim>.85f){int dr=(ag->arch+1+lcg(&ag->rng)%3)%4;ag->role[dr]+=(lcgf(&ag->rng)-.5f)*.008f;}}
    ag->energy*=.999f;
    for(int r=0;r<4;r++){if(ag->role[r]<0)ag->role[r]=0;if(ag->role[r]>1)ag->role[r]=1;}
    if(t==pt){ag->x=lcgf(&ag->rng);ag->y=lcgf(&ag->rng);ag->vx=ag->vy=0;ag->tip_val=0;}
}

__global__ void tick_c(Agent*a,Resource*r,int na,int nr,int t,int pt){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;Agent*ag=&a[i];float det=.05f,grab=.03f;float bd=det;int br=-1;for(int j=0;j<nr;j++){if(r[j].collected)continue;float dx=r[j].x-ag->x,dy=r[j].y-ag->y,d=sqrtf(dx*dx+dy*dy);if(d<bd){bd=d;br=j;}}if(br>=0&&bd<grab){r[br].collected=1;ag->res_held++;ag->energy=fminf(1,ag->energy+r[br].value*.1f);ag->fitness+=r[br].value;}else if(br>=0){float dx=r[br].x-ag->x,dy=r[br].y-ag->y,d=sqrtf(dx*dx+dy*dy);ag->vx=ag->vx*.8f+(dx/d)*.014f;ag->vy=ag->vy*.8f+(dy/d)*.014f;}else{ag->vx=ag->vx*.95f+(lcgf(&ag->rng)-.5f)*.008f;ag->vy=ag->vy*.95f+(lcgf(&ag->rng)-.5f)*.008f;}ag->x=fmodf(ag->x+ag->vx+1,1);ag->y=fmodf(ag->y+ag->vy+1,1);for(int k=0;k<32;k++){int j=lcg(&ag->rng)%na;if(j==i)continue;float dx=a[j].x-ag->x,dy=a[j].y-ag->y;if(sqrtf(dx*dx+dy*dy)>=.06f)continue;if(sqrtf(dx*dx+dy*dy)<.02f){ag->vx-=dx*.01f;ag->vy-=dy*.01f;}}ag->energy*=.999f;if(t==pt){ag->energy*=.5f;ag->x=lcgf(&ag->rng);ag->y=lcgf(&a->rng);ag->vx=ag->vy=0;}}

int main(){
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Experiment: Instinct-Driven vs Role-Driven Behavior\n");
    printf("  Based on SuperInstance/flux-instinct\n");
    printf("  8 instincts, NO energy costs, priority-based dispatch\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    Agent*da,*ha,*dc,*hc;Resource*dr;
    cudaMalloc(&da,NA*sizeof(Agent));cudaMalloc(&dc,NA*sizeof(Agent));cudaMalloc(&dr,NR*sizeof(Resource));
    ha=(Agent*)malloc(NA*sizeof(Agent));hc=(Agent*)malloc(NA*sizeof(Agent));
    int blk=(NA+255)/256,rblk=(NR+255)/256;
    
    float f_instinct=0,f_role=0,f_ctrl=0;
    for(int e=0;e<5;e++){
        // Instinct-driven
        init_a<<<blk,256>>>(da,NA);init_r<<<rblk,256>>>(dr,NR);cudaDeviceSynchronize();
        for(int t=0;t<MAXT;t++){tick_instinct<<<blk,256>>>(da,dr,NA,NR,t,250);cudaDeviceSynchronize();}
        cudaMemcpy(ha,da,NA*sizeof(Agent),cudaMemcpyDeviceToHost);float f=0;for(int i=0;i<NA;i++)f+=ha[i].fitness;f_instinct+=f;
        // Role-driven (baseline)
        init_a<<<blk,256>>>(da,NA);init_r<<<rblk,256>>>(dr,NR);cudaDeviceSynchronize();
        for(int t=0;t<MAXT;t++){tick_s<<<blk,256>>>(da,dr,NA,NR,t,250);cudaDeviceSynchronize();}
        cudaMemcpy(ha,da,NA*sizeof(Agent),cudaMemcpyDeviceToHost);f=0;for(int i=0;i<NA;i++)f+=ha[i].fitness;f_role+=f;
        // Control
        init_c<<<blk,256>>>(dc,NA);init_r<<<rblk,256>>>(dr,NR);cudaDeviceSynchronize();
        for(int t=0;t<MAXT;t++){tick_c<<<blk,256>>>(dc,dr,NA,NR,t,250);cudaDeviceSynchronize();}
        cudaMemcpy(hc,dc,NA*sizeof(Agent),cudaMemcpyDeviceToHost);f=0;for(int i=0;i<NA;i++)f+=hc[i].fitness;f_ctrl+=f;
    }
    f_instinct/=5;f_role/=5;f_ctrl/=5;
    printf("  Instinct-driven: %.1f (%.2fx ctrl)\n",f_instinct,(f_ctrl>.01)?f_instinct/f_ctrl:1);
    printf("  Role-driven:     %.1f (%.2fx ctrl)\n",f_role,(f_ctrl>.01)?f_role/f_ctrl:1);
    printf("  Control:         %.1f\n",f_ctrl);
    printf("\n─── Verdict ───\n");
    float ratio=(f_role>.01)?f_instinct/f_role:1;
    if(ratio>1.05)printf("  INSTINCT WINS: %.0f%% advantage\n",(ratio-1)*100);
    else if(ratio<.95)printf("  ROLE-DRIVEN WINS: %.0f%% advantage\n",(1/ratio-1)*100);
    else printf("  NO SIGNIFICANT DIFFERENCE (%.1f%%)\n",fabsf(ratio-1)*100);
    printf("═══════════════════════════════════════════════════════\n");
    cudaFree(da);cudaFree(dc);cudaFree(dr);free(ha);free(hc);return 0;
}
