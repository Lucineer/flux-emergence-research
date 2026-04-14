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

// RATE-OF-CHANGE NAVIGATION: Use deltas as the primary decision signal
// Mode 0: Static features (current room state only)
// Mode 1: Delta features (room score - previous room score)
// Mode 2: Delta magnitude (absolute change, regardless of direction)
// Mode 3: Velocity (trend over last 3 rooms)
// Mode 4: Momentum (keep going in direction of improvement)

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int amode[A];
__device__ int prev_score[A]; // score of previous room
__device__ int prev_room[A];
__device__ int hist[A][4]; // last 4 room scores for velocity
__device__ int momentum_dir[A]; // 0=none, 1=better, -1=worse

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__device__ int room_score(int room){
    int s=0;
    if(rt[room]==0)s-=10; // lava
    else if(rt[room]==2)s+=5; // healing
    else if(rt[room]==4)s-=3; // swamp
    for(int j=0;j<IT;j++)if(ri[room][j]>0)s+=ri[room][j];
    s+=rg[room]/5;
    return s;
}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%NA;
    prev_score[i]=room_score(ar[i]);prev_room[i]=ar[i];
    for(int h=0;h<4;h++)hist[i][h]=prev_score[i];
    momentum_dir[i]=0;
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    // Update history
    for(int h=3;h>0;h--)hist[i][h]=hist[i][h-1];
    hist[i][0]=room_score(room);

    int be=0;float bs=-999;
    for(int e=0;e<EX;e++){
        int nr=re[room][e];float sc=0;
        
        switch(mode){
            case 0: // Static: standard heuristic
                {int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;
                sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
                break;
            case 1: // Delta: compare to previous room
                {int cs=room_score(nr);
                int delta=cs-prev_score[i];
                sc=delta*2.0f;} // positive delta = good move
                break;
            case 2: // Delta magnitude: any change is interesting
                {int cs=room_score(nr);
                int delta=abs(cs-prev_score[i]);
                sc=delta*1.0f;}
                break;
            case 3: // Velocity: trend over last 3 rooms
                {int cs=room_score(nr);
                float vel=(hist[i][0]-hist[i][1])*1.5f+(hist[i][1]-hist[i][2])*0.8f;
                sc=cs*0.1f+vel*2.0f;}
                break;
            case 4: // Momentum: keep going if improving, reverse if declining
                {int cs=room_score(nr);
                if(momentum_dir[i]>=0) sc=cs*0.5f+(cs-prev_score[i])*1.5f;
                else sc=cs*0.5f-(cs-prev_score[i])*0.5f; // reverse: prefer going back
                }
                break;
        }
        
        if(sc>bs){bs=sc;be=e;}
    }
    
    int nw=re[room][be];
    int ns=room_score(nw);
    int delta=ns-room_score(room);
    if(delta>0)momentum_dir[i]=1;else if(delta<0)momentum_dir[i]=-1;
    
    prev_score[i]=room_score(room);
    prev_room[i]=room;
    ar[i]=nw;
    
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Rate-of-Change Navigation Heuristics ===\n");
    printf("128 rooms, 256 agents (52 each), 500 steps, 64 trials\n\n");
    const char* nm[]={"Static","Delta","Delta-Mag","Velocity","Momentum"};
    float tot[NA][3]={0};
    int nb=(A+BLK-1)/BLK,fb=(R+BLK-1)/BLK;
    for(int t=0;t<64;t++){
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
    printf("Mode          | Score  | Surv%% | SxS\n");
    printf("--------------+--------+-------+------\n");
    for(int m=0;m<NA;m++) printf("%-13s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nvs Static:\n");
    for(int m=1;m<NA;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
