#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define SZ 256
#define NA 512
#define STEPS 2000
#define BLK 128
#define NF 2000
#define NM 3

__device__ int fx[NF],fy[NF],fal[NF];
__device__ int ax[NA],ay[NA],ag[NA],ah[NA],aal[NA],aseed[NA];
__device__ int afleet[NA];
__device__ int gmode; // global mode for this trial

__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int mn(int a,int b){return a<b?a:b;}

__global__ void init_food(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;
    fx[i]=cr(&seed)%SZ;fy[i]=cr(&seed)%SZ;fal[i]=1;
}

__global__ void init_agents(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;afleet[i]=(i<256)?0:1;
    // Fleet A spawns top-left, Fleet B spawns bottom-right
    if(afleet[i]==0){ax[i]=cr(&aseed[i])%(SZ/2);ay[i]=cr(&aseed[i])%(SZ/2);}
    else{ax[i]=SZ/2+cr(&aseed[i])%(SZ/2);ay[i]=SZ/2+cr(&aseed[i])%(SZ/2);}
    ag[i]=0;ah[i]=100;aal[i]=1;
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA||!aal[i])return;
    int mode=gmode;int fleet=afleet[i];
    int cx=ax[i],cy=ay[i];
    
    // Find nearest food
    int bd=999,bfx=-1,bfy=-1;
    for(int f=0;f<NF;f++){
        if(!fal[f])continue;
        int dx=(fx[f]-cx+SZ)%SZ;if(dx>SZ/2)dx=SZ-dx;
        int dy=(fy[f]-cy+SZ)%SZ;if(dy>SZ/2)dy=SZ-dy;
        int d=dx+dy;if(d<bd){bd=d;bfx=fx[f];bfy=fy[f];}
    }
    
    if(bfx>=0){
        int dx=(bfx-cx+SZ)%SZ;if(dx>SZ/2)dx=dx-SZ;
        int dy=(bfy-cy+SZ)%SZ;if(dy>SZ/2)dy=dy-SZ;
        if(abs(dx)>=abs(dy)){cx=(cx+(dx?dx/abs(dx):0)+SZ)%SZ;}
        else{cy=(cy+(dy?dy/abs(dy):0)+SZ)%SZ;}
    }
    ax[i]=cx;ay[i]=cy;
    
    // Grab food
    for(int f=0;f<NF;f++){
        if(!fal[f])continue;
        if(fx[f]==cx&&fy[f]==cy){fal[f]=0;ag[i]++;break;}
    }
    
    // Fleet interactions (proximity check)
    int enemies=0,allies=0;
    for(int a=0;a<NA;a++){
        if(a==i||!aal[a])continue;
        int dx=(ax[a]-cx+SZ)%SZ;if(dx>SZ/2)dx=SZ-dx;
        int dy=(ay[a]-cy+SZ)%SZ;if(dy>SZ/2)dy=SZ-dy;
        if(dx+dy>3)continue; // interaction range
        if(afleet[a]!=fleet)enemies++;else allies++;
    }
    
    if(mode==1) ah[i]-=enemies*3; // territorial: damage from proximity
    if(mode==2) ag[i]+=allies; // cooperative: food sharing
    
    if(ah[i]<=0)aal[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;
    if(s%50==0&&!fal[i]){fal[i]=1;fx[i]=cr(&fx[i])%SZ;fy[i]=cr(&fy[i])%SZ;}}

int main(){
    printf("=== Competitive Fleets: A vs B ===\n");
    printf("256x256, 512 agents (2 fleets), 2000 food, 2000 steps, 32 trials\n");
    printf("Fleet A: top-left spawn, Fleet B: bottom-right spawn\n\n");
    
    const char* nm[]={"Peaceful","Territorial","Cooperative"};
    float tot[NM][2][3]={0}; // [mode][fleet][score,surv,sxs]
    int nb=(NA+BLK-1)/BLK,fb=(NF+BLK-1)/BLK;
    
    for(int m=0;m<NM;m++){
        int mode_h=m;
        cudaMemcpyToSymbol(gmode,&mode_h,sizeof(int),0,cudaMemcpyHostToDevice);
        
        for(int t=0;t<32;t++){
            init_food<<<fb,BLK>>>(t*999);
            init_agents<<<nb,BLK>>>(t*777);
            cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){
                step<<<nb,BLK>>>();
                if(s%200==0)cudaDeviceSynchronize();
                regen<<<fb,BLK>>>(s);
            }
            cudaDeviceSynchronize();
            
            int lg[NA],lh[NA],la[NA],lf[NA];
            cudaMemcpyFromSymbol(lg,ag,sizeof(int)*NA);
            cudaMemcpyFromSymbol(la,aal,sizeof(int)*NA);
            cudaMemcpyFromSymbol(lf,afleet,sizeof(int)*NA);
            
            float fa[2]={0},fs[2]={0};int cn[2]={0};
            for(int i=0;i<NA;i++){int f=lf[i];fa[f]+=lg[i];fs[f]+=la[i];cn[f]++;}
            for(int f=0;f<2;f++){
                tot[m][f][0]+=fa[f]/cn[f];
                tot[m][f][1]+=fs[f]/cn[f];
                tot[m][f][2]+=fa[f]/cn[f]*fs[f]/cn[f];
            }
        }
    }
    
    printf("Mode        | Fleet | Score  | Surv%% | SxS\n");
    printf("------------+-------+--------+-------+-----\n");
    for(int m=0;m<NM;m++)for(int f=0;f<2;f++)
        printf("%-11s | %s     | %6.1f | %5.1f | %.0f\n",nm[m],f?"B":"A",
            tot[m][f][0]/32,tot[m][f][1]/32*100,tot[m][f][2]/32/100);
    
    printf("\nFleet balance (A vs B):\n");
    for(int m=0;m<NM;m++){
        float ratio=tot[m][0][0]/tot[m][1][0];
        printf("  %s: A/B ratio = %.3f (1.0 = balanced)\n",nm[m],ratio);
    }
    
    printf("\nTerritorial cost:\n");
    printf("  A survival: %.1f%% (peaceful) → %.1f%% (territorial)\n",
        tot[0][0][1]/32*100,tot[1][0][1]/32*100);
    printf("  B survival: %.1f%% (peaceful) → %.1f%% (territorial)\n",
        tot[0][1][1]/32*100,tot[1][1][1]/32*100);
    
    return 0;
}
