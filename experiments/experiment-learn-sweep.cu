#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define R 128
#define A 256
#define STEPS 500
#define BLK 128
#define IT 8
#define EX 4
#define NA 6

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ float a_lat[A][4],a_w[A][16];
__device__ int amode[A];

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
}

__global__ void init_a(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A)return;
    aseed[i]=seed+i*137;ar[i]=cr(&aseed[i])%R;ah[i]=100;ag[i]=10;as[i]=0;al[i]=1;
    amode[i]=i%NA;
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

    int be=0;float bs=-999;float pred[4];
    for(int e=0;e<EX;e++){
        int nr=re[room][e];float sc=0;
        if(mode==0){int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
        else{
            float nl[4];enc(nr,nl);
            for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=a_w[i][r*4+c]*a_lat[i][c];pred[r]=fminf(1.0f,fmaxf(-1.0f,pred[r]));}
            float lr;int updates;
            switch(mode){
                case 1: sc=nl[1]*2.0f+nl[2]*0.5f-nl[3]*3.0f;lr=0.01f;updates=1;break;
                case 2: sc=nl[1]*2.0f+nl[2]*0.5f-nl[3]*3.0f;lr=0.01f;updates=10;break;
                case 3: sc=nl[1]*2.0f+nl[2]*0.5f-nl[3]*3.0f;lr=0.05f;updates=1;break;
                case 4: sc=nl[1]*2.0f+nl[2]*0.5f-nl[3]*3.0f;lr=0.05f;updates=10;break;
                case 5: sc=pred[1]*3.0f+pred[2]*2.0f-pred[3]*5.0f;lr=0.01f;updates=1;break;
            }
            if(sc>bs){bs=sc;be=e;}
            // Apply updates to chosen exit's encoding
            if(mode>=1&&mode<=4){
                float act[4];enc(re[room][be],act);
                for(int u=0;u<updates;u++)
                    for(int r=0;r<4;r++){float e=act[r]-pred[r];for(int c=0;c<4;c++)a_w[i][r*4+c]+=lr*e*a_lat[i][c];}
                for(int r=0;r<4;r++)a_lat[i][r]=act[r];
            }else if(mode==5){
                float act[4];enc(re[room][be],act);
                for(int r=0;r<4;r++){float e=act[r]-pred[r];for(int c=0;c<4;c++)a_w[i][r*4+c]+=lr*e*a_lat[i][c];}
                for(int r=0;r<4;r++)a_lat[i][r]=act[r];
            }
        }
    }
    int nw=re[room][be];ar[i]=nw;
    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Learning Rate & Update Frequency Sweep ===\n");
    printf("256 agents (43 each), 128 rooms, 500 steps, 64 trials\n\n");
    const char* nm[]={"Hardcoded","LR=0.01x1","LR=0.01x10","LR=0.05x1","LR=0.05x10","Pred-Guided"};
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
    printf("\nvs Hardcoded:\n");
    for(int m=1;m<NA;m++) printf("  %s: %+.1f%% score, %+.1f%% surv\n",nm[m],(tot[m][0]/tot[0][0]-1)*100,(tot[m][1]/tot[0][1]-1)*100);
    return 0;
}
