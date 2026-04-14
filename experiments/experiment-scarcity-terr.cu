#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define SZ 256
#define NA 512
#define STEPS 2000
#define BLK 128
#define NF 2000

// RESOURCE SCARCITY × COMPETITION: Does scarcity make territorial behavior rational?
// If food is scarce, maybe fighting IS worth it?
// Food levels: 500 (scarce), 1000 (low), 2000 (normal), 4000 (abundant)

__device__ int fx[NF],fy[NF],fal[NF];
__device__ int ax[NA],ay[NA],ag[NA],ah[NA],aal[NA],aseed[NA];
__device__ int afleet[NA];
__device__ int gmode;

__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_food(int seed,int nf){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf)return;
    fx[i]=cr(&seed)%SZ;fy[i]=cr(&seed)%SZ;fal[i]=1;
    if(i>=nf){fx[i]=0;fy[i]=0;fal[i]=0;} // zero excess
}

__global__ void init_agents(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA)return;
    aseed[i]=seed+i*137;afleet[i]=(i<256)?0:1;
    if(afleet[i]==0){ax[i]=cr(&aseed[i])%(SZ/2);ay[i]=cr(&aseed[i])%(SZ/2);}
    else{ax[i]=SZ/2+cr(&aseed[i])%(SZ/2);ay[i]=SZ/2+cr(&aseed[i])%(SZ/2);}
    ag[i]=0;ah[i]=100;aal[i]=1;
}

__global__ void step(int nf){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=NA||!aal[i])return;
    int mode=gmode;int fleet=afleet[i];
    int cx=ax[i],cy=ay[i];
    
    int bd=999,bfx=-1,bfy=-1;
    for(int f=0;f<nf;f++){
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
    
    for(int f=0;f<nf;f++){if(!fal[f])continue;if(fx[f]==cx&&fy[f]==cy){fal[f]=0;ag[i]++;break;}}
    
    int enemies=0;
    for(int a=0;a<NA;a++){
        if(a==i||!aal[a]||afleet[a]==fleet)continue;
        int dx=(ax[a]-cx+SZ)%SZ;if(dx>SZ/2)dx=SZ-dx;
        int dy=(ay[a]-cy+SZ)%SZ;if(dy>SZ/2)dy=SZ-dy;
        if(dx+dy<=3)enemies++;
    }
    
    if(mode==1) ah[i]-=enemies*3;
    if(ah[i]<=0)aal[i]=0;
}

__global__ void regen(int s,int nf){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf)return;
    if(s%50==0&&!fal[i]){fal[i]=1;fx[i]=cr(&fx[i])%SZ;fy[i]=cr(&fy[i])%SZ;}}

int main(){
    printf("=== Scarcity × Territorial Interaction ===\n");
    printf("256x256, 512 agents (2 fleets), 2000 steps, 32 trials\n\n");
    
    int food_levels[4]={500,1000,2000,4000};
    const char* fl_nm[]={"500","1000","2000","4000"};
    int modes[2]={0,1};
    const char* m_nm[]={"Peaceful","Territorial"};
    
    // [food_level][mode][fleet][metric]
    float tot[4][2][2][3]={0};
    int nb=(NA+BLK-1)/BLK;
    
    for(int fl=0;fl<4;fl++){
        int nf=food_levels[fl];
        int fb=(nf+BLK-1)/BLK;
        
        for(int m=0;m<2;m++){
            cudaMemcpyToSymbol(gmode,&m,sizeof(int),0,cudaMemcpyHostToDevice);
            
            for(int t=0;t<32;t++){
                // Reset food array first
                cudaMemset(fx,0,sizeof(int)*NF);cudaMemset(fy,0,sizeof(int)*NF);cudaMemset(fal,0,sizeof(int)*NF);
                init_food<<<fb,BLK>>>(t*999+fl*7,nf);
                init_agents<<<nb,BLK>>>(t*777);
                cudaDeviceSynchronize();
                for(int s=0;s<STEPS;s++){
                    step<<<nb,BLK>>>(nf);
                    if(s%200==0)cudaDeviceSynchronize();
                    regen<<<fb,BLK>>>(s,nf);
                }
                cudaDeviceSynchronize();
                
                int lg[NA],la[NA],lf[NA];
                cudaMemcpyFromSymbol(lg,ag,sizeof(int)*NA);
                cudaMemcpyFromSymbol(la,aal,sizeof(int)*NA);
                cudaMemcpyFromSymbol(lf,afleet,sizeof(int)*NA);
                
                float fa[2]={0},fs[2]={0};int cn[2]={0};
                for(int i=0;i<NA;i++){int f=lf[i];fa[f]+=lg[i];fs[f]+=la[i];cn[f]++;}
                for(int f=0;f<2;f++){
                    tot[fl][m][f][0]+=fa[f]/cn[f];
                    tot[fl][m][f][1]+=fs[f]/cn[f];
                    tot[fl][m][f][2]+=fa[f]/cn[f]*fs[f]/cn[f];
                }
            }
        }
    }
    
    printf("Food | Mode        | Fleet | Score  | Surv%%\n");
    printf("-----+-------------+-------+--------+-------\n");
    for(int fl=0;fl<4;fl++)for(int m=0;m<2;m++)for(int f=0;f<2;f++)
        printf("%-4s | %-11s | %s     | %6.1f | %5.1f\n",fl_nm[fl],m_nm[m],f?"B":"A",
            tot[fl][m][f][0]/32,tot[fl][m][f][1]/32*100);
    
    printf("\nTerritorial survival by food level:\n");
    for(int fl=0;fl<4;fl++)
        printf("  %s food: %.1f%% (A) %.1f%% (B)\n",fl_nm[fl],
            tot[fl][1][0][1]/32*100,tot[fl][1][1][1]/32*100);
    
    printf("\nTerritorial vs Peaceful score ratio by food level:\n");
    for(int fl=0;fl<4;fl++){
        float peace=(tot[fl][0][0][0]+tot[fl][0][1][0])/64;
        float terr=(tot[fl][1][0][0]+tot[fl][1][1][0])/64;
        printf("  %s food: Terr/Peace = %.3f\n",fl_nm[fl],terr/peace);
    }
    
    return 0;
}
