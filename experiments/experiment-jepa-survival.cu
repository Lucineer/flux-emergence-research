#include <stdio.h>
#include <cuda_runtime.h>
#define ROOMS 128
#define AGENTS 256
#define STEPS 500
#define BLK 128
#define ITEMS 8
#define EXITS 4

__device__ int rm_ter[ROOMS],rm_it[ROOMS][ITEMS],rm_ex[ROOMS][EXITS],rm_gold[ROOMS];
__device__ int ag_room[AGENTS],ag_hp[AGENTS],ag_gold[AGENTS],ag_score[AGENTS],ag_alive[AGENTS],ag_seed[AGENTS];
__device__ float ag_latent[AGENTS][4],ag_w[AGENTS][16];
__device__ int ag_deaths[AGENTS]; // track deaths for avoidance learning
__device__ float ag_danger[ROOMS]; // accumulated death memory per room

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__device__ void encode_room(int room,float* out){
    out[0]=rm_ter[room]/4.0f;
    int r=0;for(int i=0;i<ITEMS;i++)if(rm_it[room][i]>0)r++;
    out[1]=r/8.0f;
    out[2]=rm_gold[room]>0?logf(rm_gold[room]+1.0f)/5.0f:0.0f;
    int c=0;for(int a=0;a<AGENTS;a++)if(ag_alive[a]&&ag_room[a]==room)c++;
    out[3]=fminf(c/10.0f,1.0f);
}

__global__ void init_rooms(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ROOMS)return;
    rm_ter[i]=cr(&seed)%5;rm_gold[i]=cr(&seed)%50+1;
    for(int j=0;j<ITEMS;j++)rm_it[i][j]=cr(&seed)%15;
    rm_ex[i][0]=(i+1)%ROOMS;rm_ex[i][1]=(i-1+ROOMS)%ROOMS;
    rm_ex[i][2]=(i+ROOMS/2)%ROOMS;rm_ex[i][3]=(i-ROOMS/2+ROOMS)%ROOMS;
}

__global__ void init_agents(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=AGENTS)return;
    ag_seed[i]=seed+i*137;ag_room[i]=cr(&ag_seed[i])%ROOMS;
    ag_hp[i]=100;ag_gold[i]=10;ag_score[i]=0;ag_alive[i]=1;ag_deaths[i]=0;
    encode_room(ag_room[i],ag_latent[i]);
    for(int r=0;r<4;r++)for(int c=0;c<4;c++)
        ag_w[i][r*4+c]=(r==c)?0.8f+cr(&ag_seed[i])%40/100.0f:cr(&ag_seed[i])%20/100.0f-0.1f;
}

__global__ void clear_danger(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ROOMS)return;ag_danger[i]=0;}

// Mode 0: vanilla JEPA, Mode 1: JEPA + death memory, Mode 2: JEPA + danger encoding, Mode 3: hardcoded
__global__ void step_mud(int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=AGENTS||!ag_alive[i])return;
    int room=ag_room[i];
    
    for(int j=0;j<ITEMS;j++){
        if(rm_it[room][j]>0){int t=rm_it[room][j];rm_it[room][j]=0;
        if(t<=5)ag_hp[i]=mn(ag_hp[i]+t*5,100);else if(t<=10)ag_gold[i]+=t*3;else ag_score[i]+=t*2;}
    }
    int take=mn(rm_gold[room],10);ag_gold[i]+=take;rm_gold[room]-=take;ag_score[i]+=take;
    if(rm_ter[room]==0)ag_hp[i]-=2;if(rm_ter[room]==4)ag_hp[i]-=1;if(rm_ter[room]==2)ag_hp[i]=mn(ag_hp[i]+1,100);

    int best_exit=0;float best_score=-999;
    for(int e=0;e<EXITS;e++){
        int nr=rm_ex[room][e];
        float nr_lat[4];encode_room(nr,nr_lat);
        float score;
        
        if(mode==3){
            // Hardcoded
            int r=0;for(int j=0;j<ITEMS;j++)if(rm_it[nr][j]>0)r++;
            score=r*2.0f+rm_gold[nr]*0.1f-rm_ter[nr]*0.5f;
        }else{
            // JEPA prediction
            float pred[4];
            for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=ag_w[i][r*4+c]*ag_latent[i][c];pred[r]=fminf(1.0f,fmaxf(0.0f,pred[r]));}
            score=pred[1]*2.0f+nr_lat[2]*0.5f;
            
            if(mode==1){
                // Death memory: penalize rooms where deaths occurred
                score-=ag_danger[nr]*0.5f;
            }
            if(mode==2){
                // Danger encoding: weight the danger latent dim more heavily
                score-=nr_lat[3]*5.0f; // strongly avoid crowded rooms
            }
        }
        if(score>best_score){best_score=score;best_exit=e;}
    }

    int new_room=rm_ex[room][best_exit];ag_room[i]=new_room;

    // JEPA learning
    if(mode<3){
        float actual[4];encode_room(new_room,actual);
        float lr=0.01f;float pred[4];
        for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=ag_w[i][r*4+c]*ag_latent[i][c];}
        for(int r=0;r<4;r++)for(int c=0;c<4;c++)
            ag_w[i][r*4+c]+=lr*(actual[r]-pred[r])*ag_latent[i][c];
        for(int r=0;r<4;r++)ag_latent[i][r]=actual[r];
    }

    int crowd=0;for(int a=0;a<AGENTS;a++)if(a!=i&&ag_alive[a]&&ag_room[a]==new_room)crowd++;
    if(crowd>5)ag_hp[i]-=(crowd-5);
    
    if(ag_hp[i]<=0){
        ag_alive[i]=0;ag_deaths[i]++;
        if(mode==1) atomicAdd(&ag_danger[new_room],1.0f); // record death location
    }
}

__global__ void regen(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ROOMS)return;
    if(step%20==0){
        for(int j=0;j<ITEMS;j++)if(rm_it[i][j]==0&&cr(&rm_ter[i])%10<2)rm_it[i][j]=cr(&rm_ter[i])%15+1;
        rm_gold[i]=mn(rm_gold[i]+cr(&rm_ter[i])%5+1,50);
    }
}

int main(){
    printf("=== JEPA Survival Fix: Death Memory vs Danger Encoding ===\n");
    printf("128 rooms, 256 agents, 500 steps, 64 trials\n\n");
    
    const char* names[]={"JEPA-base","JEPA+death-mem","JEPA+danger-enc","Hardcoded"};
    float totals[4][3]={0}; // score, hp, survival
    
    for(int mode=0;mode<4;mode++){
        for(int trial=0;trial<64;trial++){
            clear_danger<<<1,BLK>>>();cudaDeviceSynchronize();
            init_rooms<<<1,BLK>>>(trial*999+mode*1000);cudaDeviceSynchronize();
            init_agents<<<2,BLK>>>(trial*777);cudaDeviceSynchronize();
            for(int s=0;s<STEPS;s++){step_mud<<<2,BLK>>>(mode);regen<<<1,BLK>>>(s);if(s%100==0)cudaDeviceSynchronize();}
            cudaDeviceSynchronize();
            int hp[AGENTS],sc[AGENTS],al[AGENTS];
            cudaMemcpyFromSymbol(hp,ag_hp,sizeof(int)*AGENTS);
            cudaMemcpyFromSymbol(sc,ag_score,sizeof(int)*AGENTS);
            cudaMemcpyFromSymbol(al,ag_alive,sizeof(int)*AGENTS);
            for(int i=0;i<AGENTS;i++){totals[mode][0]+=sc[i];totals[mode][1]+=hp[i];totals[mode][2]+=al[i];}
        }
    }
    
    printf("Mode               | Score  | Survival%% | Score*Surv\n");
    printf("-------------------+--------+-----------+----------\n");
    for(int m=0;m<4;m++){
        printf("%-18s | %6.1f | %9.1f | %8.0f\n",
            names[m],totals[m][0]/64/AGENTS,totals[m][2]/64/AGENTS*100,
            totals[m][0]/64*totals[m][2]/64/AGENTS);
    }
    
    // Find best overall
    int best=0;float bv=0;
    for(int m=0;m<4;m++){float v=totals[m][0]*totals[m][2];if(v>bv){bv=v;best=m;}}
    printf("\nBest overall (score × survival): %s\n",names[best]);
    
    return 0;
}
