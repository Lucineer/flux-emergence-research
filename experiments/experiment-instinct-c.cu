#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define NA 4096
#define FOOD 400
#define SZ 256
#define STEPS 3000
#define BLK 128
#define GUILDS 8

#define I_SURVIVE 1
#define I_FLEE    2
#define I_CURIOUS 3
#define I_COOP    4
#define I_GUARD   5
#define I_HOARD   7

#define T_INSTINCT 0
#define T_ROLE     1
#define T_HYBRID  2
#define T_RANDOM   3

__device__ int ax[NA],ay[NA],aalive[NA],atype[NA],acol[NA],aseed[NA];
__device__ float ahp[NA];
__device__ int fx[FOOD],fy[FOOD],falive[FOOD];
__device__ float ftimer[FOOD];
__device__ int dcs_x[GUILDS],dcs_y[GUILDS],dcs_v[GUILDS];
__device__ float pc=0.002f,gr=12.0f;

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%SZ)+SZ)%SZ;}
__device__ int td(int x1,int y1,int x2,int y2){
    int dx=x1-x2;if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    int dy=y1-y2;if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    return dx*dx+dy*dy;
}
__device__ int ei(int id,float e,float th){
    if(e<0.05f)return I_SURVIVE;if(th>0.9f)return I_FLEE;
    if(e<0.20f)return I_HOARD;if(e>0.5f)return I_GUARD;
    if(aseed[id]%10==0)return I_CURIOUS;if(th>0.7f)return I_COOP;
    return I_GUARD;
}

__global__ void init_w(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;ax[i]=rn(&aseed[i])%SZ;ay[i]=rn(&aseed[i])%SZ;
    ahp[i]=1.0f;aalive[i]=1;acol[i]=0;
    if(i<NA/4)atype[i]=T_INSTINCT;else if(i<NA/2)atype[i]=T_ROLE;
    else if(i<3*NA/4)atype[i]=T_HYBRID;else atype[i]=T_RANDOM;
}
__global__ void init_f(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    int s=seed+i*777;fx[i]=s%SZ;fy[i]=(s*31)%SZ;falive[i]=1;ftimer[i]=0;
}
__global__ void do_resp(int unused){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=FOOD)return;
    if(!falive[i]){ftimer[i]+=1.0f;if(ftimer[i]>50.0f){falive[i]=1;ftimer[i]=0;}}
}
__device__ void mt(int i,int tx,int ty){
    int dx=tx-ax[i],dy=ty-ay[i];
    if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx);ay[i]=to(ay[i]+dy);}
}
__device__ void ms(int i,int tx,int ty){
    int dx=tx-ax[i],dy=ty-ay[i];
    if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
}
__device__ void dm(int i){
    int g=i%GUILDS;if(!dcs_v[g])return;
    int dx=dcs_x[g]-ax[i],dy=dcs_y[g]-ay[i];
    if(dx<-SZ/2)dx+=SZ;if(dx>SZ/2)dx-=SZ;
    if(dy<-SZ/2)dy+=SZ;if(dy>SZ/2)dy-=SZ;
    if(dx!=0||dy!=0){ax[i]=to(ax[i]+dx/2);ay[i]=to(ay[i]+dy/2);}
}
__device__ int tg(int i,int bf,int bd){
    if(bf<0||bd>(int)(gr*gr))return 0;
    if(falive[bf]){falive[bf]=0;acol[i]++;int g=i%GUILDS;
    dcs_x[g]=fx[bf];dcs_y[g]=fy[bf];dcs_v[g]=1;return 1;}return 0;
}
__device__ void seek(int i,int bf,int bd,int ud){
    if(!tg(i,bf,bd)){if(ud)dm(i);else if(bf>=0)mt(i,fx[bf],fy[bf]);}
}
__global__ void ss(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NA||!aalive[i])return;
    float e=ahp[i];int g=i%GUILDS;
    int bd=999999,bf=-1;
    if(e>pc){for(int f=0;f<FOOD;f++){if(!falive[f])continue;
    int d=td(ax[i],ay[i],fx[f],fy[f]);if(d<bd){bd=d;bf=f;}}e-=pc;}
    int ud=0;if(dcs_v[g]){int dd=td(ax[i],ay[i],dcs_x[g],dcs_y[g]);
    if(dd<(int)(gr*gr*4)&&bd>dd)ud=1;}
    if(atype[i]==T_INSTINCT){
        float th=(float)bd/(float)(SZ*SZ)*4.0f;int ins=ei(i,e,th);
        if(ins==I_SURVIVE){e+=0.01f;}
        else if(ins==I_FLEE){if(bf>=0){ax[i]=to(ax[i]-(fx[bf]-ax[i])/2);ay[i]=to(ay[i]-(fy[bf]-ay[i])/2);}}
        else if(ins==I_HOARD){if(!tg(i,bf,bd)){if(bf>=0)ms(i,fx[bf],fy[bf]);else if(ud)dm(i);}}
        else if(ins==I_CURIOUS){ax[i]=to(ax[i]+rn(&aseed[i])%5-2);ay[i]=to(ay[i]+rn(&aseed[i])%5-2);}
        else if(ins==I_COOP){if(ud)dm(i);else seek(i,bf,bd,0);}
        else seek(i,bf,bd,ud);
    }else if(atype[i]==T_ROLE){
        if(e<0.05f){e+=0.01f;}else seek(i,bf,bd,ud);
    }else if(atype[i]==T_HYBRID){
        float th=(float)bd/(float)(SZ*SZ)*4.0f;int ins=ei(i,e,th);
        if(ins==I_SURVIVE){e+=0.01f;}else seek(i,bf,bd,ud);
    }else{
        ax[i]=to(ax[i]+rn(&aseed[i])%7-3);ay[i]=to(ay[i]+rn(&aseed[i])%7-3);
        if(bf>=0&&bd<=(int)(gr*gr))tg(i,bf,bd);
    }
    e-=0.001f;if(e<=0.0f)aalive[i]=0;ahp[i]=e;
}
int main(){
    printf("=== Instinct-C v2: Equal Survival, Isolating Instinct Overhead ===\n");
    printf("4096 agents, 400 food, 3000 steps\n");
    printf("All types stop+recover at e<0.05\n\n");
    srand(time(NULL));int seed=rand();
    int bk=(NA+BLK-1)/BLK,fb=(FOOD+BLK-1)/BLK;
    int z[GUILDS];for(int i=0;i<GUILDS;i++)z[i]=0;
    cudaMemcpyToSymbol(dcs_v,z,sizeof(int)*GUILDS);
    cudaEvent_t st,en;cudaEventCreate(&st);cudaEventCreate(&en);cudaEventRecord(st);
    init_w<<<bk,BLK>>>(seed);
    init_f<<<fb,BLK>>>(seed+999);
    cudaDeviceSynchronize();
    for(int s=0;s<STEPS;s++){
        ss<<<bk,BLK>>>(s);
        do_resp<<<fb,BLK>>>(0);
        if(s%500==0)cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    cudaEventRecord(en);cudaEventSynchronize(en);float ms;cudaEventElapsedTime(&ms,st,en);
    int hc[NA],ha[NA],ht[NA];
    cudaMemcpyFromSymbol(hc,acol,sizeof(int)*NA);
    cudaMemcpyFromSymbol(ha,aalive,sizeof(int)*NA);
    cudaMemcpyFromSymbol(ht,atype,sizeof(int)*NA);
    int tc[4]={0},ta[4]={0};
    for(int i=0;i<NA;i++){tc[ht[i]]+=hc[i];ta[ht[i]]+=ha[i];}
    const char*nm[]={"INSTINCT","ROLE","HYBRID","RANDOM"};
    printf("=== Per-Agent Results ===\n");
    float bp=0;int bt=0;
    for(int t=0;t<4;t++){
        float pa=(float)tc[t]/(NA/4);
        printf("%-10s: %7d total  %3d alive  %7.1f/agent\n",nm[t],tc[t],ta[t],pa);
        if(pa>bp){bp=pa;bt=t;}
    }
    printf("\nWinner: %s (%.1f/agent)\n",nm[bt],bp);
    for(int t=0;t<4;t++){if(t!=bt)printf("  vs %-10s: %.2fx\n",nm[t],bp/((float)tc[t]/(NA/4)));}
    printf("\nIf ROLE wins -> instinct is pure overhead (Law 2 extended)\n");
    printf("If HYBRID wins -> instinct adds value as safety net\n");
    printf("If INSTINCT wins -> complex decision-making is optimal\n");
    printf("\nTime: %.1fms (%.0f us/step)\n",ms,ms*1000/STEPS);
    cudaEventDestroy(st);cudaEventDestroy(en);return 0;
}
