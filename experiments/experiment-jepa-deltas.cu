#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define A 256
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ int aprev_r[A],aprev_h[A],aprev_g[A];
__device__ int r_agent_d[R],r_gold_d[R],r_item_d[R],r_death_d[R];
__device__ float r_hp_d[R];
__device__ float a_lat[A][4],a_w[A][16],a_sur[A];
__device__ int amode[A]; // 0=hard, 1=static, 2=delta, 3=delta+surprise

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}
__device__ int to(int v){return((v%R)+R)%R;}

__device__ void enc_static(int room,float*o){
    o[0]=rt[room]/4.0f;int r=0;for(int i=0;i<IT;i++)if(ri[room][i]>0)r++;
    o[1]=r/8.0f;o[2]=rg[room]>0?logf(rg[room]+1.0f)/5.0f:0.0f;
    int c=0;for(int a=0;a<A;a++)if(al[a]&&ar[a]==room)c++;o[3]=fminf(c/10.0f,1.0f);
}

__device__ void enc_delta(int room,float*o){
    o[0]=r_agent_d[room]>0?fminf(r_agent_d[room]/5.0f,1.0f):fmaxf(r_agent_d[room]/5.0f,-1.0f);
    o[1]=r_gold_d[room]>0?fminf(r_gold_d[room]/20.0f,1.0f):fmaxf(r_gold_d[room]/20.0f,-1.0f);
    o[2]=fminf(r_item_d[room]/4.0f,1.0f);
    o[3]=r_death_d[room]>0?1.0f:fminf(r_hp_d[room]/30.0f,1.0f);
}

__global__ void init_r(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    rt[i]=cr(&seed)%5;rg[i]=cr(&seed)%50+1;
    for(int j=0;j<IT;j++)ri[i][j]=cr(&seed)%15;
    re[i][0]=(i+1)%R;re[i][1]=(i-1+R)%R;re[i][2]=(i+R/2)%R;re[i][3]=(i-R/2+R)%R;
    r_agent_d[i]=0;r_gold_d[i]=0;r_item_d[i]=0;r_hp_d[i]=0;r_death_d[i]=0;
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    aprev_r[i]=ar[i];aprev_h[i]=100;aprev_g[i]=10;
    amode[i]=i%4;a_sur[i]=0;
    float sl[4];enc_static(ar[i],sl);
    for(int r=0;r<4;r++)for(int c=0;c<4;c++)
        a_w[i][r*4+c]=(r==c)?0.8f+cr(&aseed[i])%40/100.0f:cr(&aseed[i])%20/100.0f-0.1f;
    for(int r=0;r<4;r++)a_lat[i][r]=sl[r];
}

__global__ void clear_d(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    r_agent_d[i]=0;r_gold_d[i]=0;r_item_d[i]=0;r_hp_d[i]=0;r_death_d[i]=0;}

__global__ void rec_d(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int c=ar[i],p=aprev_r[i];
    if(c!=p){atomicAdd(&r_agent_d[c],1);atomicAdd(&r_agent_d[p],-1);}
    int gc=ag[i]-aprev_g[i];if(gc>0)atomicAdd(&r_gold_d[ar[i]],gc);
    int hl=aprev_h[i]-ah[i];if(hl>0)atomicAdd(&r_hp_d[ar[i]],hl);
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    int mode=amode[i];aprev_r[i]=ar[i];aprev_h[i]=ah[i];aprev_g[i]=ag[i];
    if(!al[i])return;
    int room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        atomicAdd(&r_item_d[room],1);
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);

    int be=0;float bs=-999;
    float pred[4];
    for(int e=0;e<EX;e++){
        int nr=re[room][e];float sc=0;
        if(mode==0){int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
        else{
            float dl[4];enc_delta(nr,dl);
            for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=a_w[i][r*4+c]*a_lat[i][c];pred[r]=fminf(1.0f,fmaxf(-1.0f,pred[r]));}
            if(mode==1){float sl[4];enc_static(nr,sl);sc=sl[1]*2.0f+sl[2]*0.5f-sl[3]*3.0f;}
            else if(mode==2){sc=dl[1]*3.0f+dl[2]*2.0f-dl[3]*8.0f;if(dl[0]>0.5f)sc-=2.0f;}
            else{
                float err=0;for(int r=0;r<4;r++){float e=dl[r]-pred[r];err+=e*e;}
                a_sur[i]=a_sur[i]*0.9f+err*0.1f;
                sc=dl[1]*3.0f+dl[2]*2.0f-dl[3]*8.0f+ a_sur[i]*2.0f;
                if(dl[3]>0.5f)sc-=a_sur[i]*5.0f;
            }
        }
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    if(mode>=1){float act[4];enc_delta(nw,act);
        float lr=0.02f;
        for(int r=0;r<4;r++){float e=act[r]-pred[r];for(int c=0;c<4;c++)a_w[i][r*4+c]+=lr*e*a_lat[i][c];}
        for(int r=0;r<4;r++)a_lat[i][r]=act[r];}
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0){al[i]=0;atomicAdd(&r_death_d[nw],1);}
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Delta JEPA: Rate-of-Change Perception ===\n");
    printf("128 rooms, 256 agents (64 each), 500 steps, 64 trials\n\n");
    const char* nm[]={"Hardcoded","Static-JEPA","Delta-JEPA","Delta+Surprise"};
    float tot[4][3]={0};
    for(int t=0;t<64;t++){
        init_r<<<1,BLK>>>(t*999);init_a<<<2,BLK>>>(t*777);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){clear_d<<<1,BLK>>>();
            step<<<2,BLK>>>();cudaDeviceSynchronize();rec_d<<<2,BLK>>>();regen<<<1,BLK>>>(s);if(s%100==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();
        int hp[A],sc[A],aa[A],md[A];
        cudaMemcpyFromSymbol(hp,ah,sizeof(int)*A);cudaMemcpyFromSymbol(sc,as,sizeof(int)*A);
        cudaMemcpyFromSymbol(aa,al,sizeof(int)*A);cudaMemcpyFromSymbol(md,amode,sizeof(int)*A);
        float ts[4]={0},ta[4]={0};int cn[4]={0};
        for(int i=0;i<A;i++){int m=md[i];ts[m]+=sc[i];ta[m]+=aa[i];cn[m]++;}
        for(int m=0;m<4;m++){tot[m][0]+=ts[m]/cn[m];tot[m][1]+=ta[m]/cn[m];tot[m][2]+=ts[m]/cn[m]*ta[m]/cn[m];}
    }
    printf("Mode          | Score  | Surv%% | SxS\n");
    printf("--------------+--------+-------+------\n");
    for(int m=0;m<4;m++) printf("%-13s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nvs Hardcoded:\n");
    for(int m=1;m<4;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
