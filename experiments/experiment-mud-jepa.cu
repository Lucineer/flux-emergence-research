#include <stdio.h>
#include <cuda_runtime.h>

// Mini-JEPA MUD Experiment
// Can a tiny learned perception model (8 params per agent) improve MUD survival?
// Compare: hardcoded perception vs JEPA-tiny prediction

#define ROOMS 128
#define EXITS 4
#define ITEMS 8
#define AGENTS 256
#define STEPS 500
#define BLK 128

// Room structure
__device__ int rm_ter[ROOMS];       // terrain type (0-4)
__device__ int rm_it[ROOMS][ITEMS]; // item types (0=empty, 1-20)
__device__ int rm_ex[ROOMS][EXITS]; // exit room indices
__device__ int rm_gold[ROOMS];      // gold available

// Agent state
__device__ int ag_room[AGENTS];
__device__ int ag_hp[AGENTS];
__device__ int ag_gold[AGENTS];
__device__ int ag_score[AGENTS];
__device__ int ag_alive[AGENTS];
__device__ int ag_seed[AGENTS];

// JEPA latent: 4-dim embedding of room state
// Encodes: terrain, item richness, danger level, crowding
__device__ float ag_latent[AGENTS][4]; // current room embedding
__device__ float ag_pred[AGENTS][4];    // predicted next room embedding
__device__ float ag_w[AGENTS][16];      // JEPA weights (4x4 matrix, learned by evolution)

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int mx(int a,int b){return a>b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

// Encode room into 4-dim latent
__device__ void encode_room(int room, float* out, int* seed){
    // dim0: terrain value (0.0-1.0)
    out[0] = rm_ter[room] / 4.0f;
    // dim1: item richness (count of non-empty items / 8)
    int rich=0; for(int i=0;i<ITEMS;i++) if(rm_it[room][i]>0) rich++;
    out[1] = rich / 8.0f;
    // dim2: gold value (log scale)
    out[2] = rm_gold[room] > 0 ? logf(rm_gold[room]+1.0f)/5.0f : 0.0f;
    // dim3: danger (count of other agents in this room / 10)
    int crowd=0; for(int a=0;a<AGENTS;a++) if(ag_alive[a]&&ag_room[a]==room) crowd++;
    out[3] = fminf(crowd/10.0f, 1.0f);
}

// JEPA forward: predict next room embedding from current
__device__ void jepa_predict(float* latent, float* w, float* pred){
    for(int i=0;i<4;i++){
        pred[i]=0;
        for(int j=0;j<4;j++) pred[i]+=w[i*4+j]*latent[j];
        pred[i]=fminf(1.0f,fmaxf(0.0f,pred[i])); // sigmoid-like clamp
    }
}

// JEPA action: choose exit based on predicted latents
__device__ int jepa_choose_exit(int room, float* pred, int* seed){
    // Score each exit by how well its predicted latent matches desired state
    // Desired: high richness (dim1), low danger (dim3)
    float best_score=-999; int best_exit=0;
    for(int e=0;e<EXITS;e++){
        int nr=rm_ex[room][e];
        float nr_lat[4]; encode_room(nr,nr_lat,seed);
        // Score = predicted_richness - predicted_danger + gold_bonus
        float score = pred[1]*2.0f - pred[3]*3.0f + nr_lat[2]*0.5f;
        if(score>best_score){best_score=score;best_exit=e;}
    }
    return best_exit;
}

// Hardcoded action: greedy heuristic
__device__ int hardcoded_choose_exit(int room, int* seed){
    // Move to room with most items and gold
    float best_score=-999; int best_exit=0;
    for(int e=0;e<EXITS;e++){
        int nr=rm_ex[room][e];
        int rich=0; for(int i=0;i<ITEMS;i++) if(rm_it[nr][i]>0) rich++;
        float score = rich*2.0f + rm_gold[nr]*0.1f - rm_ter[nr]*0.5f;
        if(score>best_score){best_score=score;best_exit=e;}
    }
    return best_exit;
}

__global__ void init_rooms(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ROOMS)return;
    rm_ter[i]=cr(&seed)%5;
    rm_gold[i]=cr(&seed)%50+1;
    for(int j=0;j<ITEMS;j++) rm_it[i][j]=cr(&seed)%15;
    rm_ex[i][0]=(i+1)%ROOMS; rm_ex[i][1]=(i-1+ROOMS)%ROOMS;
    rm_ex[i][2]=(i+ROOMS/2)%ROOMS; rm_ex[i][3]=(i-ROOMS/2+ROOMS)%ROOMS;
}

__global__ void init_agents(int seed, int use_jepa){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=AGENTS)return;
    ag_seed[i]=seed+i*137;
    ag_room[i]=cr(&ag_seed[i])%ROOMS;
    ag_hp[i]=100; ag_gold[i]=10; ag_score[i]=0; ag_alive[i]=1;
    if(use_jepa){
        encode_room(ag_room[i],ag_latent[i],&ag_seed[i]);
        jepa_predict(ag_latent[i],ag_w[i],ag_pred[i]);
        // Init weights: identity-ish + some exploration bias
        for(int r=0;r<4;r++) for(int c=0;c<4;c++)
            ag_w[i][r*4+c]=(r==c)?0.8f+cr(&ag_seed[i])%40/100.0f:cr(&ag_seed[i])%20/100.0f-0.1f;
    }
}

__global__ void step_mud(int use_jepa){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=AGENTS||!ag_alive[i])return;
    int room=ag_room[i];

    // Collect items in current room
    for(int j=0;j<ITEMS;j++){
        if(rm_it[room][j]>0){
            int type=rm_it[room][j];
            rm_it[room][j]=0;
            if(type>=1&&type<=5) ag_hp[i]=mn(ag_hp[i]+type*5, 100);
            else if(type>=6&&type<=10) ag_gold[i]+=type*3;
            else if(type>=11&&type<=15) ag_score[i]+=type*2;
        }
    }
    // Collect gold
    int take=mn(rm_gold[room],10);
    ag_gold[i]+=take; rm_gold[room]-=take;
    ag_score[i]+=take;

    // Terrain effects
    if(rm_ter[room]==0) ag_hp[i]-=2; // desert
    if(rm_ter[room]==4) ag_hp[i]-=1; // swamp
    if(rm_ter[room]==2) ag_hp[i]=mn(ag_hp[i]+1,100); // forest heals

    // Move
    int exit;
    if(use_jepa){
        jepa_predict(ag_latent[i],ag_w[i],ag_pred[i]);
        exit=jepa_choose_exit(room,ag_pred[i],&ag_seed[i]);
    }else{
        exit=hardcoded_choose_exit(room,&ag_seed[i]);
    }
    int new_room=rm_ex[room][exit];
    ag_room[i]=new_room;

    // Update JEPA latent
    if(use_jepa){
        float actual[4]; encode_room(new_room,actual,&ag_seed[i]);
        // Simple weight update: nudge toward better prediction
        float lr=0.01f;
        for(int r=0;r<4;r++)
            for(int c=0;c<4;c++)
                ag_w[i][r*4+c]+=lr*(actual[r]-ag_pred[i][r])*ag_latent[i][c];
        // Update latent
        for(int r=0;r<4;r++) ag_latent[i][r]=actual[r];
    }

    // Agent collision damage (crowded rooms)
    int crowd=0;
    for(int a=0;a<AGENTS;a++) if(a!=i&&ag_alive[a]&&ag_room[a]==new_room) crowd++;
    if(crowd>5) ag_hp[i]-=(crowd-5);

    // Death check
    if(ag_hp[i]<=0) ag_alive[i]=0;
}

// Regenerate items slowly
__global__ void regen(int step){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ROOMS)return;
    if(step%20==0){
        for(int j=0;j<ITEMS;j++) if(rm_it[i][j]==0&&cr(&rm_ter[i])%10<2) rm_it[i][j]=cr(&rm_ter[i])%15+1;
        rm_gold[i]=mn(rm_gold[i]+cr(&rm_ter[i])%5+1, 50);
    }
}

int main(){
    printf("=== Mini-JEPA MUD Experiment ===\n");
    printf("128 rooms, 256 agents, 500 steps, 64 trials\n\n");
    printf("Mode       | Avg Score | Avg HP  | Survival%% | Avg Gold\n");
    printf("------------+-----------+---------+-----------+---------\n");

    for(int trial=0;trial<64;trial++){
        float jepa_score=0,jepa_hp=0,jepa_surv=0,jepa_gold=0;
        float hard_score=0,hard_hp=0,hard_surv=0,hard_gold=0;

        // Run JEPA agents
        init_rooms<<<1,BLK>>>(trial*999);cudaDeviceSynchronize();
        init_agents<<<2,BLK>>>(trial*777,1);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){step_mud<<<2,BLK>>>(1);regen<<<1,BLK>>>(s);if(s%100==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();
        int h_hp[AGENTS],h_sc[AGENTS],h_al[AGENTS],h_go[AGENTS];
        cudaMemcpyFromSymbol(h_hp,ag_hp,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(h_sc,ag_score,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(h_al,ag_alive,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(h_go,ag_gold,sizeof(int)*AGENTS);
        for(int i=0;i<AGENTS;i++){jepa_score+=h_sc[i];jepa_hp+=h_hp[i];jepa_surv+=h_al[i];jepa_gold+=h_go[i];}

        // Run hardcoded agents
        init_rooms<<<1,BLK>>>(trial*999);cudaDeviceSynchronize();
        init_agents<<<2,BLK>>>(trial*777,0);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){step_mud<<<2,BLK>>>(0);regen<<<1,BLK>>>(s);if(s%100==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(h_hp,ag_hp,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(h_sc,ag_score,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(h_al,ag_alive,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(h_go,ag_gold,sizeof(int)*AGENTS);
        for(int i=0;i<AGENTS;i++){hard_score+=h_sc[i];hard_hp+=h_hp[i];hard_surv+=h_al[i];hard_gold+=h_go[i];}

        if(trial%16==0){
            printf("Trial %d\n",trial);
            printf("  JEPA:      %7.1f | %7.1f | %9.1f | %7.1f\n",
                jepa_score/AGENTS,jepa_hp/AGENTS,jepa_surv/AGENTS*100,jepa_gold/AGENTS);
            printf("  Hardcoded: %7.1f | %7.1f | %9.1f | %7.1f\n",
                hard_score/AGENTS,hard_hp/AGENTS,hard_surv/AGENTS*100,hard_gold/AGENTS);
        }
    }
    return 0;
}
