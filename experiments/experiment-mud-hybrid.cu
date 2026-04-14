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
__device__ int ag_mode[AGENTS]; // 0=hardcoded, 1=jepa, 2=hybrid

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
    ag_hp[i]=100;ag_gold[i]=10;ag_score[i]=0;ag_alive[i]=1;
    ag_mode[i]=i%3; // 0=hardcoded, 1=jepa, 2=hybrid
    encode_room(ag_room[i],ag_latent[i]);
    for(int r=0;r<4;r++)for(int c=0;c<4;c++)
        ag_w[i][r*4+c]=(r==c)?0.8f+cr(&ag_seed[i])%40/100.0f:cr(&ag_seed[i])%20/100.0f-0.1f;
}

__global__ void step_mud(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=AGENTS||!ag_alive[i])return;
    int room=ag_room[i],mode=ag_mode[i];
    
    // Collect
    for(int j=0;j<ITEMS;j++){
        if(rm_it[room][j]>0){
            int t=rm_it[room][j];rm_it[room][j]=0;
            if(t<=5)ag_hp[i]=mn(ag_hp[i]+t*5,100);
            else if(t<=10)ag_gold[i]+=t*3;
            else ag_score[i]+=t*2;
        }
    }
    int take=mn(rm_gold[room],10);ag_gold[i]+=take;rm_gold[room]-=take;ag_score[i]+=take;
    if(rm_ter[room]==0)ag_hp[i]-=2;
    if(rm_ter[room]==4)ag_hp[i]-=1;
    if(rm_ter[room]==2)ag_hp[i]=mn(ag_hp[i]+1,100);

    // Move - 3 strategies
    int best_exit=0;float best_score=-999;
    for(int e=0;e<EXITS;e++){
        int nr=rm_ex[room][e];
        float nr_lat[4];encode_room(nr,nr_lat);
        float score;
        if(mode==0){
            // Hardcoded: items + gold - terrain
            int r=0;for(int j=0;j<ITEMS;j++)if(rm_it[nr][j]>0)r++;
            score=r*2.0f+rm_gold[nr]*0.1f-rm_ter[nr]*0.5f;
        }else if(mode==1){
            // JEPA: predict and score
            float pred[4];
            for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=ag_w[i][r*4+c]*ag_latent[i][c];pred[r]=fminf(1.0f,fmaxf(0.0f,pred[r]));}
            score=pred[1]*2.0f-pred[3]*3.0f+nr_lat[2]*0.5f;
        }else{
            // HYBRID: JEPA for richness + hardcoded safety
            float pred[4];
            for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=ag_w[i][r*4+c]*ag_latent[i][c];pred[r]=fminf(1.0f,fmaxf(0.0f,pred[r]));}
            // JEPA drives toward richness
            float jepa_score=pred[1]*3.0f+nr_lat[2]*0.5f;
            // Hardcoded safety: penalize crowded rooms
            int crowd=0;for(int a=0;a<AGENTS;a++)if(ag_alive[a]&&ag_room[a]==nr)crowd++;
            float safety=(crowd>5)?-10.0f:(crowd>3)?-2.0f:0.0f;
            // Hardcoded terrain avoidance
            float terrain_pen=(rm_ter[nr]==0)?-3.0f:(rm_ter[nr]==4)?-2.0f:(rm_ter[nr]==2)?1.0f:0.0f;
            score=jepa_score+safety+terrain_pen;
        }
        if(score>best_score){best_score=score;best_exit=e;}
    }

    int new_room=rm_ex[room][best_exit];ag_room[i]=new_room;

    // JEPA learning (for mode 1 and 2)
    if(mode>=1){
        float actual[4];encode_room(new_room,actual);
        float lr=0.01f;float pred[4];
        for(int r=0;r<4;r++){pred[r]=0;for(int c=0;c<4;c++)pred[r]+=ag_w[i][r*4+c]*ag_latent[i][c];}
        for(int r=0;r<4;r++)for(int c=0;c<4;c++)
            ag_w[i][r*4+c]+=lr*(actual[r]-pred[r])*ag_latent[i][c];
        for(int r=0;r<4;r++)ag_latent[i][r]=actual[r];
    }

    // Collision damage
    int crowd=0;for(int a=0;a<AGENTS;a++)if(a!=i&&ag_alive[a]&&ag_room[a]==new_room)crowd++;
    if(crowd>5)ag_hp[i]-=(crowd-5);
    if(ag_hp[i]<=0)ag_alive[i]=0;
}

__global__ void regen(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=ROOMS)return;
    if(step%20==0){
        for(int j=0;j<ITEMS;j++)if(rm_it[i][j]==0&&cr(&rm_ter[i])%10<2)rm_it[i][j]=cr(&rm_ter[i])%15+1;
        rm_gold[i]=mn(rm_gold[i]+cr(&rm_ter[i])%5+1,50);
    }
}

int main(){
    printf("=== MUD Perception Showdown: Hardcoded vs JEPA vs Hybrid ===\n");
    printf("128 rooms, 256 agents (85 each), 500 steps, 64 trials\n");
    printf("Hybrid = JEPA richness perception + hardcoded safety\n\n");
    printf("Trial | Hardcoded Score/Surv | JEPA Score/Surv | Hybrid Score/Surv\n");
    printf("------+--------------------+------------------+-------------------\n");

    float h_sc[3]={0},h_hp[3]={0},h_sv[3]={0},h_go[3]={0};
    for(int trial=0;trial<64;trial++){
        init_rooms<<<1,BLK>>>(trial*999);init_agents<<<2,BLK>>>(trial*777);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){step_mud<<<2,BLK>>>();regen<<<1,BLK>>>(s);if(s%100==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();
        int hp[AGENTS],sc[AGENTS],al[AGENTS],go[AGENTS],md[AGENTS];
        cudaMemcpyFromSymbol(hp,ag_hp,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(sc,ag_score,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(al,ag_alive,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(go,ag_gold,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(md,ag_mode,sizeof(int)*AGENTS);
        float ts[3]={0},th[3]={0},ta[3]={0},tg[3]={0};
        int cnt[3]={0};
        for(int i=0;i<AGENTS;i++){int m=md[i];ts[m]+=sc[i];th[m]+=hp[i];ta[m]+=al[i];tg[m]+=go[i];cnt[m]++;}
        for(int m=0;m<3;m++){h_sc[m]+=ts[m]/cnt[m];h_hp[m]+=th[m]/cnt[m];h_sv[m]+=ta[m]/cnt[m];h_go[m]+=tg[m]/cnt[m];}

        if(trial%16==0){
            printf("%5d | %5.0f/%4.0f%%      | %5.0f/%4.0f%%     | %5.0f/%4.0f%%\n",
                trial,ts[0]/cnt[0],ta[0]/cnt[0]*100,ts[1]/cnt[1],ta[1]/cnt[1]*100,
                ts[2]/cnt[2],ta[2]/cnt[2]*100);
        }
    }
    printf("\n=== FINAL AVERAGES (64 trials) ===\n");
    printf("Mode       | Score  | HP    | Survival%% | Gold\n");
    printf("------------+--------+-------+-----------+-------\n");
    printf("Hardcoded  | %6.1f | %5.1f | %9.1f | %5.1f\n",h_sc[0]/64,h_hp[0]/64,h_sv[0]/64*100,h_go[0]/64);
    printf("JEPA       | %6.1f | %5.1f | %9.1f | %5.1f\n",h_sc[1]/64,h_hp[1]/64,h_sv[1]/64*100,h_go[1]/64);
    printf("Hybrid     | %6.1f | %5.1f | %9.1f | %5.1f\n",h_sc[2]/64,h_hp[2]/64,h_sv[2]/64*100,h_go[2]/64);
    printf("\nBest score:  %s\n",h_sc[1]>h_sc[0]&&h_sc[1]>h_sc[2]?"JEPA":h_sc[2]>h_sc[0]?"Hybrid":"Hardcoded");
    printf("Best survival: %s\n",h_sv[0]>h_sv[1]&&h_sv[0]>h_sv[2]?"Hardcoded":h_sv[2]>h_sv[1]?"Hybrid":"JEPA");
    printf("Best overall: %s\n",h_sc[2]*h_sv[2]>h_sc[0]*h_sv[0]&&h_sc[2]*h_sv[2]>h_sc[1]*h_sv[1]?"Hybrid":
        h_sc[0]*h_sv[0]>h_sc[1]*h_sv[1]?"Hardcoded":"JEPA");
    return 0;
}
