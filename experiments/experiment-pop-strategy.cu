#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define MAXA 512
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4

// POPULATION DENSITY × STRATEGY: How does the best strategy change with fewer agents?
// Populations: 16, 32, 64, 128, 256, 512
// Strategies: Static, Delta-Penalty, Death-Obs, Stigmergy-Read, Crowding-Avoid

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[MAXA],ah[MAXA],ag[MAXA],as[MAXA],al[MAXA],aseed[MAXA];
__device__ int amode[MAXA];
__device__ int death_count[R];
__device__ int stig_s[R];

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
    death_count[i]=0;stig_s[i]=0;
}

__global__ void init_a(int seed,int na){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%5;
}

__global__ void step(int na){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    float prev_sc=0;{int r=0;for(int j=0;j<IT;j++)if(ri[room][j]>0)r++;prev_sc=r*2.0f+rg[room]*0.1f-rt[room]*0.5f;}

    int be=0;float bs=-999;
    for(int e=0;e<EX;e++){
        int nr=re[room][e];
        int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;
        float sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;
        
        // Count agents in target room
        int crowd=0;for(int a=0;a<na;a++)if(a!=i&&al[a]&&ar[a]==nr)crowd++;
        
        switch(mode){
            case 0: break; // pure static
            case 1: // delta penalty
                if(sc<prev_sc)sc-=abs(sc-prev_sc)*0.5f;
                break;
            case 2: // death observation
                if(death_count[nr]>2)sc-=5.0f;
                break;
            case 3: // stigmergy read
                if(stig_s[nr]>0)sc+=stig_s[nr]*0.1f;
                break;
            case 4: // crowding avoid
                sc-=crowd*1.5f;
                break;
        }
        
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    
    // Mode 3: leave stigmergy marks
    if(mode==3){int items=0;for(int j=0;j<IT;j++)if(ri[nw][j]>0)items++;if(items>3)atomicAdd(&stig_s[nw],1);}
    
    int cr2=0;for(int a=0;a<na;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);
    if(ah[i]<=0){al[i]=0;atomicAdd(&death_count[room],1);}
}

__global__ void regen(int s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}
    if(s%100==0)death_count[i]=0;
    if(s%50==0)stig_s[i]=(stig_s[i]>0)?stig_s[i]-1:0;
}

int main(){
    printf("=== Population × Strategy Interaction ===\n");
    printf("128 rooms, 500 steps, 64 trials\n\n");
    
    int pops[]={16,32,64,128,256,512};
    const char* pnm[]={"16","32","64","128","256","512"};
    const char* mnm[]={"Static","Delta-Pen","Death-Obs","Stig-Read","Crowd-Avoid"};
    
    float tot[6][5][2]={0}; // [pop_idx][mode][score,surv]
    
    for(int p=0;p<6;p++){
        int na=pops[p];
        int nb=(na+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
        
        for(int t=0;t<64;t++){
            init_r<<<fb,BLK>>>(t*999);
            init_a<<<nb,BLK>>>(t*777,na);
            cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){step<<<nb,BLK>>>(na);if(s%100==0)cudaDeviceSynchronize();regen<<<fb,BLK>>>(s);}
            cudaDeviceSynchronize();
            
            int lg[MAXA],la[MAXA],md[MAXA];
            cudaMemcpyFromSymbol(lg,ag,sizeof(int)*MAXA);
            cudaMemcpyFromSymbol(la,al,sizeof(int)*MAXA);
            cudaMemcpyFromSymbol(md,amode,sizeof(int)*MAXA);
            
            float ts[5]={0},ta[5]={0};int cn[5]={0};
            for(int i=0;i<na;i++){int m=md[i]%5;ts[m]+=lg[i];ta[m]+=la[i];cn[m]++;}
            for(int m=0;m<5;m++){tot[p][m][0]+=ts[m]/cn[m];tot[p][m][1]+=ta[m]/cn[m];}
        }
    }
    
    printf("Pop  | Strategy     | Score  | Surv%% | vs Static\n");
    printf("-----+--------------+--------+-------+----------\n");
    for(int p=0;p<6;p++){
        for(int m=0;m<5;m++){
            float vs=(tot[p][m][0]/tot[p][0][0]-1)*100;
            printf("%-4s | %-12s | %6.1f | %5.1f | %+.1f%%\n",pnm[p],mnm[m],
                tot[p][m][0]/64,tot[p][m][1]/64*100,m==0?0:vs);
        }
        printf("-----+--------------+--------+-------+----------\n");
    }
    
    printf("\nBest strategy by population:\n");
    for(int p=0;p<6;p++){
        int best=0;float bv=0;
        for(int m=1;m<5;m++){float v=tot[p][m][0];if(v>tot[p][best][0])best=m;}
        printf("  Pop %-4s: %s (%.1f score)\n",pnm[p],mnm[best],tot[p][best][0]/64);
    }
    
    return 0;
}
