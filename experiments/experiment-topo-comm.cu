#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define A 256
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4
#define NA 4

// TOPOLOGY VARIATION: Does connectivity matter?
// Test: 4 exits (standard), 6 exits (highway), 2 exits (corridor), 8 exits (hub)
// Hypothesis: More exits = communication becomes useful, fewer exits = harder to coordinate

__device__ int rt[R],ri[R][IT],re[R][8],rg[R]; // max 8 exits
__device__ int nexits[R]; // per-room exit count
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int amode[A];
__device__ int stig_s[R];

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_r(int seed,int topo){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    stig_s[i]=0;
    int ne;
    switch(topo){
        case 0: ne=2;break; // corridor
        case 1: ne=4;break; // standard
        case 2: ne=6;break; // highway
        case 3: ne=8;break; // hub
        default: ne=4;
    }
    nexits[i]=ne;
    // Generate exits: wrap-around + skip pattern
    for(int e=0;e<ne;e++){
        int offset=1;
        switch(e){
            case 0: offset=1;break;
            case 1: offset=-1;break;
            case 2: offset=R/2;break;
            case 3: offset=-R/2;break;
            case 4: offset=R/3;break;
            case 5: offset=-R/3;break;
            case 6: offset=R/4;break;
            case 7: offset=-R/4;break;
        }
        re[i][e]=(i+offset+R)%R;
    }
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%NA;
}

__global__ void step(int topo){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    int ne=nexits[room];if(ne<2)ne=2;
    int be=0;float bs=-999;
    for(int e=0;e<ne;e++){
        int nr=re[room][e];float sc=0;
        {int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
        if(mode==2){ // stigmergy
            for(int e2=0;e2<ne;e2++){int nr2=re[room][e2];if(stig_s[nr2]>0)sc+=stig_s[nr2]*0.1f;}
        }
        else if(mode==3){ // stigmergy + mark
            for(int e2=0;e2<ne;e2++){int nr2=re[room][e2];if(stig_s[nr2]>0)sc+=stig_s[nr2]*0.1f;}
        }
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    
    if(mode>=2){
        int items=0;for(int j=0;j<IT;j++)if(ri[nw][j]>0)items++;
        if(items>3)atomicAdd(&stig_s[nw],1);
    }
    
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void regen(int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}
    if(s%50==0)stig_s[i]=(stig_s[i]>0)?stig_s[i]-1:0;
}

int main(){
    printf("=== Topology × Communication Interaction ===\n");
    printf("128 rooms, 256 agents (64 each), 500 steps, 64 trials\n");
    printf("Topologies: 2-exit, 4-exit, 6-exit, 8-exit\n\n");
    const char* topo_nm[]={"2-exit","4-exit","6-exit","8-exit"};
    const char* mode_nm[]={"Hardcoded","Stig-Read","Stig-Mark","Stig-Both"};
    float tot[4][4][3]={0}; // [topo][mode][metric]
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    
    for(int topo=0;topo<4;topo++){
        for(int t=0;t<64;t++){
            init_r<<<fb,BLK>>>(t*999,topo);init_a<<<nb,BLK>>>(t*777);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){step<<<nb,BLK>>>(topo);if(s%100==0)cudaDeviceSynchronize();regen<<<fb,BLK>>>(s);}
            cudaDeviceSynchronize();
            int hp[A],sc[A],aa[A],md[A];
            cudaMemcpyFromSymbol(hp,ah,sizeof(int)*A);cudaMemcpyFromSymbol(sc,as,sizeof(int)*A);
            cudaMemcpyFromSymbol(aa,al,sizeof(int)*A);cudaMemcpyFromSymbol(md,amode,sizeof(int)*A);
            float ts[4]={0},ta[4]={0};int cn[4]={0};
            for(int i=0;i<A;i++){int m=md[i]%4;ts[m]+=sc[i];ta[m]+=aa[i];cn[m]++;}
            for(int m=0;m<4;m++){tot[topo][m][0]+=ts[m]/cn[m];tot[topo][m][1]+=ta[m]/cn[m];tot[topo][m][2]+=ts[m]/cn[m]*ta[m]/cn[m];}
        }
    }
    
    printf("Topology | Mode          | Score  | Surv%%\n");
    printf("---------+---------------+--------+-------\n");
    for(int topo=0;topo<4;topo++)
        for(int m=0;m<4;m++)
            printf("%-8s | %-13s | %6.1f | %5.1f\n",topo_nm[topo],mode_nm[m],tot[topo][m][0]/64,tot[topo][m][1]/64*100);
    
    printf("\nStigmergy lift by topology (vs Hardcoded):\n");
    for(int topo=0;topo<4;topo++){
        float hard=tot[topo][0][0]/64;
        float stig=tot[topo][3][0]/64;
        printf("  %s exits: %+.1f%% (stig vs hard)\n",topo_nm[topo],(stig/hard-1)*100);
    }
    
    printf("\nHardcoded score by topology:\n");
    for(int topo=0;topo<4;topo++){
        printf("  %s exits: %.1f (vs 4-exit baseline: %+.1f%%)\n",topo_nm[topo],tot[topo][0][0]/64,
            (tot[topo][0][0]/tot[1][0][0]-1)*100);
    }
    return 0;
}
