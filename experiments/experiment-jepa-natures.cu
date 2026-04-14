#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define ROOMS 128
#define AGENTS 320 // 64 per model (5 models)
#define STEPS 500
#define BLK 128
#define ITEMS 8
#define EXITS 4
#define WSIZE 4

__device__ int rm_ter[ROOMS],rm_it[ROOMS][ITEMS],rm_ex[ROOMS][EXITS],rm_gold[ROOMS];
__device__ int ag_room[AGENTS],ag_hp[AGENTS],ag_gold[AGENTS],ag_score[AGENTS],ag_alive[AGENTS],ag_seed[AGENTS];
__device__ float ag_latent[AGENTS][WSIZE],ag_w[AGENTS][WSIZE*WSIZE];
__device__ int ag_model[AGENTS]; // which architecture 0-4
__device__ float ag_surprise[AGENTS];

__device__ int mn(int a,int b){return a<b?a:b;}
__device__ int cr(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__device__ void encode(int room,float* out){
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

// 5 different initialization strategies
__global__ void init_agents(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=AGENTS)return;
    ag_seed[i]=seed+i*137;ag_room[i]=cr(&ag_seed[i])%ROOMS;
    ag_hp[i]=100;ag_gold[i]=10;ag_score[i]=0;ag_alive[i]=1;
    ag_model[i]=i%5;ag_surprise[i]=0;
    encode(ag_room[i],ag_latent[i]);
    
    int model=i%5;
    for(int r=0;r<WSIZE;r++)for(int c=0;c<WSIZE;c++){
        float v;
        switch(model){
            case 0: // Random uniform [-0.3, 0.3]
                v=cr(&ag_seed[i])%600/1000.0f-0.3f;break;
            case 1: // I-JEPA style: small std, mild sparsity
                v=cr(&ag_seed[i])%600/1000.0f-0.3f;
                v*=0.15f; // shrink std
                if(cr(&ag_seed[i])%10<2) v=0; // 20% sparsity
                break;
            case 2: // Xavier: std = sqrt(2/(in+out)) = sqrt(2/8) = 0.5
                v=cr(&ag_seed[i])%600/1000.0f-0.3f;
                v*=0.5f;break;
            case 3: // Sparse random: 50% zeros
                v=(cr(&ag_seed[i])%2==0)?0:cr(&ag_seed[i])%600/1000.0f-0.3f;break;
            case 4: // Safety-biased: strong negative weight on danger dim (index 3)
                v=cr(&ag_seed[i])%600/1000.0f-0.3f;
                if(c==3) v=fminf(v,-0.3f); // always penalize danger
                if(r==3&&c!=3) v=fmaxf(v,0.2f); // boost survival features
                break;
        }
        ag_w[i][r*WSIZE+c]=v;
    }
}

__global__ void step_mud(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=AGENTS||!ag_alive[i])return;
    int model=ag_model[i],room=ag_room[i];
    
    // Collect
    for(int j=0;j<ITEMS;j++){
        if(rm_it[room][j]>0){int t=rm_it[room][j];rm_it[room][j]=0;
        if(t<=5)ag_hp[i]=mn(ag_hp[i]+t*5,100);else if(t<=10)ag_gold[i]+=t*3;else ag_score[i]+=t*2;}
    }
    int take=mn(rm_gold[room],10);ag_gold[i]+=take;rm_gold[room]-=take;ag_score[i]+=take;
    if(rm_ter[room]==0)ag_hp[i]-=2;if(rm_ter[room]==4)ag_hp[i]-=1;if(rm_ter[room]==2)ag_hp[i]=mn(ag_hp[i]+1,100);

    // Predict + choose
    int best_exit=0;float best_score=-999;
    float pred[WSIZE];
    for(int e=0;e<EXITS;e++){
        int nr=rm_ex[room][e];
        float nl[WSIZE];encode(nr,nl);
        // Forward pass
        for(int r=0;r<WSIZE;r++){
            pred[r]=0;for(int c=0;c<WSIZE;c++)pred[r]+=ag_w[i][r*WSIZE+c]*ag_latent[i][c];
            pred[r]=fminf(1.0f,fmaxf(-1.0f,pred[r]));
        }
        // Score based on model's "nature"
        float score;
        switch(model){
            case 0: score=pred[1]*2.0f+nl[2]*0.5f-nl[3]*3.0f;break; // balanced
            case 1: // I-JEPA style: trust prediction more
                score=pred[1]*3.0f+pred[2]*2.0f-pred[3]*5.0f;break;
            case 2: // Xavier: use prediction + current state equally
                score=(pred[1]+nl[1])*1.5f+(pred[2]+nl[2])*0.5f-nl[3]*2.0f;break;
            case 3: // Sparse: mostly follow prediction (sparse = decisive)
                score=pred[1]*4.0f-pred[3]*6.0f;break;
            case 4: // Safety-biased: strong danger avoidance
                score=pred[1]*1.0f-pred[3]*10.0f+nl[2]*0.3f;break;
        }
        if(score>best_score){best_score=score;best_exit=e;}
    }

    int new_room=rm_ex[room][best_exit];ag_room[i]=new_room;

    // Learn
    float actual[WSIZE];encode(new_room,actual);
    float err_sum=0;
    for(int r=0;r<WSIZE;r++){
        float err=actual[r]-pred[r];err_sum+=err*err;
        float lr=0.01f;
        for(int c=0;c<WSIZE;c++) ag_w[i][r*WSIZE+c]+=lr*err*ag_latent[i][c];
    }
    ag_surprise[i]=ag_surprise[i]*0.9f+err_sum*0.1f;
    for(int r=0;r<WSIZE;r++) ag_latent[i][r]=actual[r];

    // Collision
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
    printf("=== JEPA Nature Competition: 5 Initialization Strategies ===\n");
    printf("320 agents (64 each), 128 rooms, 500 steps, 64 trials\n\n");
    printf("Model          | Init Strategy      | Score  | Surv%% | Gold  | S×Surv\n");
    printf("---------------+--------------------+--------+-------+-------+-------\n");

    const char* names[]={"Random","I-JEPA-style","Xavier","Sparse-50%%","Safety-bias"};
    float totals[5][4]={0};

    for(int trial=0;trial<64;trial++){
        init_rooms<<<1,BLK>>>(trial*999);init_agents<<<3,BLK>>>(trial*777);cudaDeviceSynchronize();
        for(int s=0;s<STEPS;s++){step_mud<<<3,BLK>>>();regen<<<1,BLK>>>(s);if(s%100==0)cudaDeviceSynchronize();}
        cudaDeviceSynchronize();
        int hp[AGENTS],sc[AGENTS],al[AGENTS],go[AGENTS],md[AGENTS];
        cudaMemcpyFromSymbol(hp,ag_hp,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(sc,ag_score,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(al,ag_alive,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(go,ag_gold,sizeof(int)*AGENTS);
        cudaMemcpyFromSymbol(md,ag_model,sizeof(int)*AGENTS);
        float ts[5]={0},th[5]={0},ta[5]={0},tg[5]={0};int cnt[5]={0};
        for(int i=0;i<AGENTS;i++){int m=md[i];ts[m]+=sc[i];th[m]+=hp[i];ta[m]+=al[i];tg[m]+=go[i];cnt[m]++;}
        for(int m=0;m<5;m++){totals[m][0]+=ts[m]/cnt[m];totals[m][1]+=th[m]/cnt[m];totals[m][2]+=ta[m]/cnt[m];totals[m][3]+=tg[m]/cnt[m];}
    }

    for(int m=0;m<5;m++){
        printf("%-14s | %-18s | %6.1f | %5.1f | %5.1f | %5.0f\n",
            names[m],names[m],totals[m][0]/64,totals[m][2]/64*100,totals[m][3]/64,
            totals[m][0]/64*totals[m][2]/64/100);
    }
    
    printf("\n=== Key Findings ===\n");
    // Find winner by score×survival
    int best=0;float bv=0;
    for(int m=0;m<5;m++){float v=totals[m][0]*totals[m][2];if(v>bv){bv=v;best=m;}}
    printf("Best overall: %s (score×surv)\n",names[best]);
    
    // Find safest
    int safest=0;for(int m=1;m<5;m++)if(totals[m][2]>totals[safest][2])safest=m;
    printf("Safest: %s (%.1f%% survival)\n",names[safest],totals[safest][2]/64);
    
    // Find highest scorer
    int scorer=0;for(int m=1;m<5;m++)if(totals[m][0]>totals[scorer][0])scorer=m;
    printf("Highest score: %s (%.1f)\n",names[scorer],totals[scorer][0]/64);
    
    return 0;
}
