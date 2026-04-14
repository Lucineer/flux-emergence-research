#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define A 256
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4
#define NA 5

// ROOM MEMORY: Can agents remember which rooms are good/bad?
// Modes: No memory, Personal memory (per-agent room scores), 
//        Shared memory (global room scores), Born-knowledge (preloaded danger map)

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int amode[A];
__device__ int pers_mem[A][R]; // per-agent room memory (-5 to +10)
__device__ int glob_mem[R]; // shared room memory
__device__ int danger_map[R]; // preloaded: which rooms are near lava

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int mx(int a,int b){return a>b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
    // Build danger map: rooms adjacent to lava are dangerous
    danger_map[i]=0;
    for(int e=0;e<EX;e++){if(rt[re[i][e]]==0)danger_map[i]=1;}
    glob_mem[i]=0;
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%NA;
    for(int r=0;r<R;r++)pers_mem[i][r]=0;
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    // Update memory based on experience
    int exp_delta=0;
    if(rt[room]==0)exp_delta=-3; // lava room = bad
    else if(rg[room]>20)exp_delta=2; // rich room = good
    else if(rg[room]<5)exp_delta=-1; // depleted = slightly bad
    else exp_delta=0;
    
    if(mode==1){ // Personal memory
        pers_mem[i][room]=mx(-5,mn(10,pers_mem[i][room]+exp_delta));
    } else if(mode==2){ // Shared memory
        atomicAdd(&glob_mem[room],exp_delta);
    } else if(mode==3){ // Personal + shared
        pers_mem[i][room]=mx(-5,mn(10,pers_mem[i][room]+exp_delta));
        atomicAdd(&glob_mem[room],exp_delta/2);
    }
    
    int be=0;float bs=-999;
    for(int e=0;e<EX;e++){
        int nr=re[room][e];float sc=0;
        {int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
        
        if(mode==1) sc+=pers_mem[i][nr]*0.3f;
        else if(mode==2) sc+=glob_mem[nr]*0.1f;
        else if(mode==3) sc+=pers_mem[i][nr]*0.2f+glob_mem[nr]*0.05f;
        else if(mode==4) sc+=danger_map[nr]*(-5.0f); // born-knowledge: avoid danger rooms
        
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Room Memory Systems in MUD ===\n");
    printf("128 rooms, 256 agents (52 each), 500 steps, 64 trials\n\n");
    const char* nm[]={"No-Memory","Personal","Shared","Personal+Shared","Born-Knowledge"};
    float tot[NA][3]={0};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    for(int t=0;t<64;t++){
        int zero=0;cudaMemcpyToSymbol(glob_mem,&zero,sizeof(int),0,cudaMemcpyHostToDevice);
        init_r<<<fb,BLK>>>(t*999);init_a<<<nb,BLK>>>(t*777);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){step<<<nb,BLK>>>();if(s%100==0)cudaDeviceSynchronize();regen<<<fb,BLK>>>(s);}
        cudaDeviceSynchronize();
        int hp[A],sc[A],aa[A],md[A];
        cudaMemcpyFromSymbol(hp,ah,sizeof(int)*A);cudaMemcpyFromSymbol(sc,as,sizeof(int)*A);
        cudaMemcpyFromSymbol(aa,al,sizeof(int)*A);cudaMemcpyFromSymbol(md,amode,sizeof(int)*A);
        float ts[NA]={0},ta[NA]={0};int cn[NA]={0};
        for(int i=0;i<A;i++){int m=md[i];ts[m]+=sc[i];ta[m]+=aa[i];cn[m]++;}
        for(int m=0;m<NA;m++){tot[m][0]+=ts[m]/cn[m];tot[m][1]+=ta[m]/cn[m];tot[m][2]+=ts[m]/cn[m]*ta[m]/cn[m];}
    }
    printf("Mode              | Score  | Surv%% | SxS\n");
    printf("------------------+--------+-------+------\n");
    for(int m=0;m<NA;m++) printf("%-17s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nvs No-Memory:\n");
    for(int m=1;m<NA;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
