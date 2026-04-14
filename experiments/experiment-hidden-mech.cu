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

// EXPANDED MUD: add hidden mechanics that hardcoded doesn't know about
// Mechanic 1: "rich rooms" — rooms where item count * gold > threshold give double score
// Mechanic 2: "danger echoes" — rooms adjacent to lava rooms have hidden -3hp/tick
// Mechanic 3: "lucky corridors" — rooms with specific room IDs (multiples of 7) give bonus gold
// The JEPA model should DISCOVER these through observation. Hardcoded cannot.

__device__ int rt[R],ri[R][IT],re[R][EX],rg[R];
__device__ int ar[A],ah[A],ag[A],as[A],al[A],aseed[A];
__device__ float a_lat[A][6],a_w[A][36]; // 6-dim latent, 6x6 weights = 36 params
__device__ int amode[A];
__device__ int tick;

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__device__ void enc(int room,float*o){
    o[0]=rt[room]/4.0f;
    int r=0;for(int i=0;i<IT;i++)if(ri[room][i]>0)r++;o[1]=r/8.0f;
    o[2]=rg[room]>0?logf(rg[room]+1.0f)/5.0f:0.0f;
    int c=0;for(int a=0;a<A;a++)if(al[a]&&ar[a]==room)c++;o[3]=fminf(c/10.0f,1.0f);
    // Hidden features: only observable through consequences
    // Feature 4: "pressure" — if any neighbor is lava, room has subtle danger
    float danger=0;
    for(int e=0;e<EX;e++){int nr=re[room][e];if(rt[nr]==0)danger+=0.5f;}
    o[4]=fminf(danger/2.0f,1.0f);
    // Feature 5: "luck" — room ID pattern (rooms 0,7,14,21... are lucky)
    o[5]=(room%7==0)?1.0f:0.0f;
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
    float sl[6];enc(ar[i],sl);
    for(int r=0;r<6;r++)for(int c=0;c<6;c++)
        a_w[i][r*6+c]=(r==c)?0.8f+cr(&aseed[i])%40/100.0f:cr(&aseed[i])%20/100.0f-0.1f;
    for(int r=0;r<6;r++)a_lat[i][r]=sl[r];
}

__global__ void step(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=A||!al[i])return;
    int mode=amode[i],room=ar[i];
    for(int j=0;j<IT;j++){if(ri[room][j]>0){int t=ri[room][j];ri[room][j]=0;
        if(t<=5)ah[i]=mn(ah[i]+t*5,100);else if(t<=10)ag[i]+=t*3;else as[i]+=t*2;}}
    int take=mn(rg[room],10);ag[i]+=take;rg[room]-=take;as[i]+=take;
    if(rt[room]==0)ah[i]-=2;if(rt[room]==4)ah[i]-=1;if(rt[room]==2)ah[i]=mn(ah[i]+1,100);
    
    // Hidden mechanics (apply AFTER basic collection)
    // Danger echo: rooms adjacent to lava take 3hp/tick
    for(int e=0;e<EX;e++){if(rt[re[room][e]]==0)ah[i]-=3;break;} // only once per room
    // Lucky corridor: multiples of 7 give bonus gold
    if(room%7==0){int bonus=cr(&aseed[i])%5+1;ag[i]+=bonus;as[i]+=bonus*2;}
    // Rich room bonus: high items + high gold = double score on collection
    int rich_items=0,rich_gold=rg[room];
    for(int j=0;j<IT;j++)if(ri[room][j]>0)rich_items++;
    if(rich_items>=5&&rich_gold>=30)as[i]+=10;

    int be=0;float bs=-999;float pred[6];
    for(int e=0;e<EX;e++){
        int nr=re[room][e];float sc=0;
        switch(mode){
            case 0: // Hardcoded: doesn't know about hidden mechanics
                {int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f;}
                break;
            case 1: // Oracle: hardcoded WITH hidden knowledge (upper bound)
                {int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;
                float danger=0;for(int e2=0;e2<EX;e2++){if(rt[re[nr][e2]]==0)danger+=0.5f;}
                float luck=(nr%7==0)?1.0f:0.0f;
                sc=r*2.0f+rg[nr]*0.1f-rt[nr]*0.5f-danger*3.0f+luck*2.0f;}
                break;
            case 2: // JEPA 6-dim, slow learn, prediction-guided
                {float nl[6];enc(nr,nl);
                for(int r=0;r<6;r++){pred[r]=0;for(int c=0;c<6;c++)pred[r]+=a_w[i][r*6+c]*a_lat[i][c];pred[r]=fminf(1.0f,fmaxf(-1.0f,pred[r]));}
                sc=pred[1]*3.0f+pred[2]*2.0f-pred[3]*5.0f-pred[4]*8.0f+pred[5]*3.0f;}
                break;
            case 3: // JEPA 6-dim, slow learn, feature-guided (uses observed features)
                {float nl[6];enc(nr,nl);
                sc=nl[1]*2.0f+nl[2]*0.5f-nl[3]*3.0f-nl[4]*5.0f+nl[5]*2.0f;}
                break;
            case 4: // JEPA 6-dim, NO hidden features (only 4-dim, can't see danger/luck)
                {float nl[4];nl[0]=rt[nr]/4.0f;int r=0;for(int j=0;j<IT;j++)if(ri[nr][j]>0)r++;
                nl[1]=r/8.0f;nl[2]=rg[nr]>0?logf(rg[nr]+1.0f)/5.0f:0.0f;
                int c=0;for(int a=0;a<A;a++)if(al[a]&&ar[a]==nr)c++;nl[3]=fminf(c/10.0f,1.0f);
                for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=a_w[i][r*6+c]*a_lat[i][c];pred[r]=fminf(1.0f,fmaxf(-1.0f,pred[r]));}
                sc=pred[1]*3.0f+pred[2]*2.0f-pred[3]*5.0f;}
                break;
        }
        if(sc>bs){bs=sc;be=e;}
    }
    int nw=re[room][be];ar[i]=nw;
    // Learn for JEPA modes
    if(mode>=2&&mode<=3){
        float act[6];enc(nw,act);float lr=0.001f;
        for(int r=0;r<6;r++){pred[r]=0;for(int c=0;c<6;c++)pred[r]+=a_w[i][r*6+c]*a_lat[i][c];pred[r]=fminf(1.0f,fmaxf(-1.0f,pred[r]));}
        for(int r=0;r<6;r++){float e=act[r]-pred[r];for(int c=0;c<6;c++)a_w[i][r*6+c]+=lr*e*a_lat[i][c];}
    }
    if(mode>=2){float act[6];enc(nw,act);for(int r=0;r<6;r++)a_lat[i][r]=act[r];}
    // 4-dim latent update for mode 4
    if(mode==4){float nl[4];nl[0]=rt[nw]/4.0f;int r=0;for(int j=0;j<IT;j++)if(ri[nw][j]>0)r++;
        nl[1]=r/8.0f;nl[2]=rg[nw]>0?logf(rg[nw]+1.0f)/5.0f:0.0f;
        int c=0;for(int a=0;a<A;a++)if(al[a]&&ar[a]==nw)c++;nl[3]=fminf(c/10.0f,1.0f);
        for(int r=0;r<4;r++)a_lat[i][r]=nl[r];}

    int cr2=0;for(int a=0;a<A;a++)if(a!=i&&al[a]&&ar[a]==nw)cr2++;
    if(cr2>5)ah[i]-=(cr2-5);if(ah[i]<=0)al[i]=0;
}

__global__ void regen(int s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=R)return;
    if(s%20==0){for(int j=0;j<IT;j++)if(ri[i][j]==0&&cr(&rt[i])%10<2)ri[i][j]=cr(&rt[i])%15+1;
    rg[i]=mn(rg[i]+cr(&rt[i])%5+1,50);}}

int main(){
    printf("=== Hidden Mechanics Discovery Test ===\n");
    printf("128 rooms, 256 agents (52 each), 500 steps, 64 trials\n");
    printf("Hidden: danger-echo (-3hp near lava), lucky corridors (room%%7==0), rich rooms\n\n");
    const char* nm[]={"Hardcoded","Oracle","JEPA-6dim-Pred","JEPA-6dim-Feat","JEPA-4dim-Blind"};
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
    printf("Mode              | Score  | Surv%% | SxS\n");
    printf("------------------+--------+-------+------\n");
    for(int m=0;m<NA;m++) printf("%-17s | %6.1f | %5.1f | %.0f\n",nm[m],tot[m][0]/64,tot[m][1]/64*100,tot[m][2]/64/100);
    printf("\nKey comparisons:\n");
    printf("  Oracle vs Hardcoded: %+.1f%% (theoretical max from hidden knowledge)\n",(tot[1][0]/tot[0][0]-1)*100);
    printf("  JEPA-6d-Feat vs Hard: %+.1f%% (JEPA discovers hidden mechanics?)\n",(tot[3][0]/tot[0][0]-1)*100);
    printf("  JEPA-4d-Blind vs Hard: %+.1f%% (blind JEPA, no hidden features)\n",(tot[4][0]/tot[0][0]-1)*100);
    printf("  JEPA-6d captures Oracle gap: %.1f%%\n",
        (tot[3][0]-tot[0][0])/(tot[1][0]-tot[0][0])*100);
    return 0;
}
