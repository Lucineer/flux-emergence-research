#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define SZ 256
#define NA 512
#define STEPS 2000
#define BLK 128
#define NF 2000

__device__ int fx[NF],fy[NF],fal[NF];
__device__ int ax[NA],ay[NA],ag[NA],ah[NA],aal[NA],aseed[NA];
__device__ int afleet[NA];
__device__ int gmode_a,gmode_b; // per-fleet mode

__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_food(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;
    fx[i]=cr(&seed)%SZ;fy[i]=cr(&seed)%SZ;fal[i]=1;
}

__global__ void init_agents(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;afleet[i]=(i<256)?0:1;
    if(afleet[i]==0){ax[i]=cr(&aseed[i])%(SZ/2);ay[i]=cr(&aseed[i])%(SZ/2);}
    else{ax[i]=SZ/2+cr(&aseed[i])%(SZ/2);ay[i]=SZ/2+cr(&aseed[i])%(SZ/2);}
    ag[i]=0;ah[i]=100;aal[i]=1;
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA||!aal[i])return;
    int fleet=afleet[i];
    int mode=(fleet==0)?gmode_a:gmode_b;
    int cx=ax[i],cy=ay[i];
    
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
    
    for(int f=0;f<NF;f++){if(!fal[f])continue;if(fx[f]==cx&&fy[f]==cy){fal[f]=0;ag[i]++;break;}}
    
    int enemies=0;
    for(int a=0;a<NA;a++){
        if(a==i||!aal[a]||afleet[a]==fleet)continue;
        int dx=(ax[a]-cx+SZ)%SZ;if(dx>SZ/2)dx=SZ-dx;
        int dy=(ay[a]-cy+SZ)%SZ;if(dy>SZ/2)dy=SZ-dy;
        if(dx+dy<=3)enemies++;
    }
    
    // mode 0=peaceful, 1=territorial(damage), 2=cooperative(food-sharing with allies only, cap)
    if(mode==1) ah[i]-=enemies*3;
    // Cooperative: no per-tick bonus, just share food info (move toward food allies found)
    // Actually let's just test peaceful vs territorial for clean results
    
    if(ah[i]<=0)aal[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NF)return;
    if(s%50==0&&!fal[i]){fal[i]=1;fx[i]=cr(&fx[i])%SZ;fy[i]=cr(&fy[i])%SZ;}}

int main(){
    printf("=== Asymmetric Fleet Competition ===\n");
    printf("256x256, 512 agents, 2000 food, 2000 steps, 32 trials\n\n");
    
    // Test: Peaceful vs Peaceful, Territorial vs Territorial, 
    //        Peaceful-A vs Territorial-B, Territorial-A vs Peaceful-B
    int configs[4][2]={{0,0},{1,1},{0,1},{1,0}};
    const char* names[]={"Peace vs Peace","Terr vs Terr","Peace-A vs Terr-B","Terr-A vs Peace-B"};
    float tot[4][2][3]={0};
    int nb=(NA+BLK-1)/BLK,fb=(NF+BLK-1)/BLK;
    
    for(int cfg=0;cfg<4;cfg++){
        cudaMemcpyToSymbol(gmode_a,&configs[cfg][0],sizeof(int),0,cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(gmode_b,&configs[cfg][1],sizeof(int),0,cudaMemcpyHostToDevice);
        
        for(int t=0;t<32;t++){
            init_food<<<fb,BLK>>>(t*999+cfg*7);
            init_agents<<<nb,BLK>>>(t*777+cfg*13);
            cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){
                step<<<nb,BLK>>>();
                if(s%200==0)cudaDeviceSynchronize();
                regen<<<fb,BLK>>>(s);
            }
            cudaDeviceSynchronize();
            
            int lg[NA],la[NA],lf[NA];
            cudaMemcpyFromSymbol(lg,ag,sizeof(int)*NA);
            cudaMemcpyFromSymbol(la,aal,sizeof(int)*NA);
            cudaMemcpyFromSymbol(lf,afleet,sizeof(int)*NA);
            
            float fa[2]={0},fs[2]={0};int cn[2]={0};
            for(int i=0;i<NA;i++){int f=lf[i];fa[f]+=lg[i];fs[f]+=la[i];cn[f]++;}
            for(int f=0;f<2;f++){
                tot[cfg][f][0]+=fa[f]/cn[f];
                tot[cfg][f][1]+=fs[f]/cn[f];
                tot[cfg][f][2]+=fa[f]/cn[f]*fs[f]/cn[f];
            }
        }
    }
    
    printf("Config                  | Fleet | Score  | Surv%% | SxS\n");
    printf("------------------------+-------+--------+-------+-----\n");
    for(int cfg=0;cfg<4;cfg++)for(int f=0;f<2;f++)
        printf("%-23s | %s     | %6.1f | %5.1f | %.0f\n",names[cfg],f?"B":"A",
            tot[cfg][f][0]/32,tot[cfg][f][1]/32*100,tot[cfg][f][2]/32/100);
    
    printf("\n=== Key Findings ===\n");
    printf("Territorial self-damage:\n");
    printf("  Terr-Terr: A=%.1f%% surv, B=%.1f%% surv\n",tot[1][0][1]/32*100,tot[1][1][1]/32*100);
    printf("Peace-Peace: A=%.1f%% surv, B=%.1f%% surv\n",tot[0][0][1]/32*100,tot[0][1][1]/32*100);
    
    printf("\nAsymmetric advantage (peaceful gets free food while territorial fights):\n");
    printf("  Peace-A vs Terr-B: A=%.1f food, B=%.1f food (ratio: %.2f)\n",
        tot[2][0][0]/32,tot[2][1][0]/32,tot[2][0][0]/tot[2][1][0]);
    printf("  Terr-A vs Peace-B: A=%.1f food, B=%.1f food (ratio: %.2f)\n",
        tot[3][0][0]/32,tot[3][1][0]/32,tot[3][0][0]/tot[3][1][0]);
    
    return 0;
}
