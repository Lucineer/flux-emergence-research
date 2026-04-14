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

// COMMUNICATION PROTOCOLS IN MUD: Can agents share room observations?
// Test: gossip (share current room), DCS (share best room), stigmergy (leave marks), silent
// Hypothesis: In MUD topology (128 rooms, 4 exits each), communication helps LESS
// than in open 2D because room topology creates bottlenecks

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ float a_lat[A][4],a_w[A][16];
__device__ int amode[A];
__device__ int gossip_r[A],gossip_i[A],gossip_g[A],gossip_t[A]; // shared info
__device__ int stig_r[R],stig_s[R]; // stigmergy marks

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__device__ void enc(int room,float*o){
    o[0]=rt[room]/4.0f;int r=0;for(int i=0;i<IT;i++)if(ri[room][i]>0)r++;
    o[1]=r/8.0f;o[2]=rg[room]>0?logf(rg[room]+1.0f)/5.0f:0.0f;
    int c=0;for(int a=0;a<A;a++)if(al[a]&&ar[a]==room)c++;o[3]=fminf(c/10.0f,1.0f);
}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
    stig_r[i]=-1;stig_s[i]=0;
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%NA;gossip_t[i]=-100;
    float sl[4];enc(ar[i],sl);
    for(int r=0;r<4;r++)for(int c=0;c<4;c++)
        a_w[i][r*4+c]=(r==c)?0.8f+cr(&aseed[i])%40/100.0f:cr(&aseed[i])%20/100.0f-0.1f;
    for(int r=0;r<4;r++)a_lat[i][r]=sl[r];
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    int be=0;float bs=-999;
    for(int e=0;e<EX;e++){
        int nr=re[room][e];float sc=0;
        // Base: hardcoded heuristic
        {int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
        
        // Communication bonuses
        if(mode==1){ // Gossip: read shared info from others in same room
            for(int a=0;a<A;a++){if(a!=i&&al[a]&&ar[a]==room&&gossip_t[a]>-50){
                // Peer's reported room has items/gold — add bonus to that direction
                // If gossip room is accessible from here, bias toward it
                for(int e2=0;e2<EX;e2++){if(re[room][e2]==gossip_r[a]){
                    sc+=gossip_i[a]*0.5f+gossip_g[a]*0.05f;break;}
            }}}
        }
        else if(mode==2){ // DCS: look for highest-value room reported
            int best_r=-1,best_v=0;
            for(int a=0;a<A;a++){if(al[a]&&gossip_t[a]>-20){
                int v=gossip_i[a]*3+gossip_g[a];
                if(v>best_v){best_v=v;best_r=gossip_r[a];}
            }}
            if(best_r>=0){for(int e2=0;e2<EX;e2++){if(re[room][e2]==best_r){sc+=best_v*0.3f;break;}}}
        }
        else if(mode==3){ // Stigmergy: read room marks
            for(int e2=0;e2<EX;e2++){int nr2=re[room][e2];
                if(stig_s[nr2]>0)sc+=stig_s[nr2]*0.1f;}
        }
        // mode 4 = silent (base only)
        
        if(sc>bs){bs=sc;be=e;}
    }
    
    int nw=re[room][be];ar[i]=nw;
    
    // Update gossip for modes 1-2
    if(mode<=2){
        int items=0;for(int j=0;j<IT;j++)if(ri[nw][j]>0)items++;
        gossip_r[i]=nw;gossip_i[i]=items;gossip_g[i]=rg[nw];gossip_t[i]=0;
    }
    // Update stigmergy for mode 3
    if(mode==3){
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
    // Decay stigmergy
    if(s%50==0)stig_s[i]=(stig_s[i]>0)?stig_s[i]-1:0;
    // Age gossip
}

__global__ void age_gossip(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;gossip_t[i]--;}

int main(){
    printf("=== Communication Protocols in MUD Topology ===\n");
    printf("128 rooms, 256 agents (52 each), 500 steps, 64 trials\n");
    printf("Modes: Gossip, DCS, Stigmergy, Silent\n\n");
    const char* nm[]={"Hardcoded","Gossip","DCS","Stigmergy","Silent"};
    float tot[NA][3]={0};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    for(int t=0;t<64;t++){
        // Reset stigmergy
        // stigmergy reset handled in init_r kernel
        init_r<<<fb,BLK>>>(t*999);init_a<<<nb,BLK>>>(t*777);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){step<<<nb,BLK>>>();age_gossip<<<nb,BLK>>>();if(s%100==0)cudaDeviceSynchronize();regen<<<fb,BLK>>>(s);}
        cudaDeviceSynchronize();
        int hp[A],sc[A],aa[A],md[A];
        cudaMemcpyFromSymbol(hp,ah,sizeof(int)*A);cudaMemcpyFromSymbol(sc,as,sizeof(int)*A);
        cudaMemcpyFromSymbol(aa,al,sizeof(int)*A);cudaMemcpyFromSymbol(md,amode,sizeof(int)*A);
        float ts[NA]={0},ta[NA]={0};int cn[NA]={0};
        for(int i=0;i<A;i++){int m=md[i];ts[m]+=sc[i];ta[m]+=aa[i];cn[m]++;}
        for(int m=0;m<NA;m++){tot[m][0]+=ts[m]/cn[m];tot[m][1]+=ta[m]/cn[m];tot[m][2]+=ts[m]/cn[m]*ta[m]/cn[m];}
    }
    printf("Mode          | Score  | Surv%% | SxS\n");
    printf("--------------+--------+-------+------\n");
    for(int m=0;m<NA;m++) printf("%-13s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nvs Hardcoded (Silent = same as Hardcoded by design):\n");
    for(int m=1;m<NA;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
