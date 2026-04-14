#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NA 1024
#define ROOMS 256
#define GENS 200
#define TURNS 200
#define BLK 128
#define SCRIPTS 32
#define ACTIONS 6
#define GUILDS 8

// Actions: 0=idle, 1=attack, 2=trade, 3=flee, 4=pickup, 5=heal
__device__ int scripts[SCRIPTS][TURNS];
__device__ int fitness[SCRIPTS];
__device__ int room_agents[ROOMS],room_gold[ROOMS],room_hp[ROOMS];
__device__ int a_hp[NA],a_gold[NA],a_room[NA],a_script[NA],a_alive[NA];
__device__ int aseed[NA];

// DCS: per-guild best room
__device__ int dcs_room[GUILDS],dcs_gold[GUILDS],dcs_v[GUILDS];

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_scripts(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=SCRIPTS*TURNS)return;
    int s=i/TURNS,t=i%TURNS;
    scripts[s][t]=rn(&seed)%(ACTIONS);
}

__global__ void init_world(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NA)return;
    aseed[i]=seed+i*137;a_hp[i]=100;a_gold[i]=10;a_room[i]=rn(&aseed[i])%ROOMS;
    a_script[i]=i%SCRIPTS;a_alive[i]=1;
}

__global__ void init_rooms(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ROOMS)return;
    room_agents[i]=NA/ROOMS;room_gold[i]=rn(&seed)%50;room_hp[i]=rn(&seed)%20;
}

__global__ void reset_world(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NA)return;
    aseed[i]=seed+i*137;a_hp[i]=100;a_gold[i]=10;a_room[i]=rn(&aseed[i])%ROOMS;
    a_script[i]=i%SCRIPTS;a_alive[i]=1;
    if(i<ROOMS){room_agents[i]=NA/ROOMS;room_gold[i]=rn(&seed)%50;room_hp[i]=rn(&seed)%20;}
    int z[GUILDS];for(int j=0;j<GUILDS;j++)z[j]=0;
    // Can't use cudaMemcpyToSymbol in device code, use direct assignment
    for(int j=0;j<GUILDS;j++)dcs_v[j]=0;
}

__global__ void sim_gen(int gen,int use_dcs){
    // Reset for new generation
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NA)return;
    if(i==0){for(int r=0;r<ROOMS;r++){room_agents[r]=0;room_gold[r]=rn(&aseed[0]+r)%50;room_hp[r]=rn(&aseed[0]+r+ROOMS)%20;}}
    
    for(int t=0;t<TURNS;t++){
        int id=blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=NA||!a_alive[id])continue;
        
        int act=scripts[a_script[id]][t];
        int rm=a_room[id];
        int g=id%GUILDS;
        
        // Check DCS: move to guild's best room
        if(use_dcs&&dcs_v[g]&&dcs_room[g]!=rm&&room_gold[dcs_room[g]]>room_gold[rm]+5){
            a_room[id]=dcs_room[g];rm=dcs_room[g];act=4; // pickup
        }
        
        if(act==0){/* idle */}
        else if(act==1){/* attack */
            if(room_hp[rm]>0){room_hp[rm]--;a_gold[id]+=2;}
            else{a_hp[id]-=5;if(a_hp[id]<=0)a_alive[id]=0;}
        }else if(act==2){/* trade */
            if(a_gold[id]>5){a_gold[id]-=5;a_hp[id]+=10;}
        }else if(act==3){/* flee */
            a_room[id]=(a_room[id]+rn(&aseed[id])%ROOMS)%ROOMS;
        }else if(act==4){/* pickup */
            if(room_gold[rm]>0){int take=min(room_gold[rm],5);room_gold[rm]-=take;a_gold[id]+=take;
            // Update DCS
            if(room_gold[rm]+take>0){dcs_room[g]=rm;dcs_gold[g]=room_gold[rm]+take;dcs_v[g]=1;}}
        }else if(act==5){/* heal */
            if(a_gold[id]>3){a_gold[id]-=3;a_hp[id]+=5;}
        }
    }
    
    // Score: gold + survival bonus
    if(i<SCRIPTS){
        fitness[i]=0;
        for(int j=i;j<NA;j+=SCRIPTS){
            if(a_alive[j])fitness[i]+=a_gold[j]+50;
            else fitness[i]+=a_gold[j];
        }
    }
}

__global__ void evolve(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=SCRIPTS)return;
    // Find best 8 scripts
    int best[8],bv[8];
    for(int j=0;j<8;j++){best[j]=j;bv[j]=fitness[j];}
    for(int j=8;j<SCRIPTS;j++){
        for(int k=0;k<8;k++){
            if(fitness[j]>bv[k]){
                for(int l=7;l>k;l--){bv[l]=bv[l-1];best[l]=best[l-1];}
                bv[k]=fitness[j];best[k]=j;break;
            }
        }
    }
    // Create new generation from top 8
    if(i<8){
        for(int j=0;j<TURNS;j++)scripts[i][j]=scripts[best[i]][j];
    }else{
        int parent=best[rn(&aseed[i])%8];
        int mut=rn(&aseed[i])%100;
        for(int j=0;j<TURNS;j++){
            scripts[i][j]=scripts[parent][j];
            if(mut<10)scripts[i][j]=rn(&aseed[i])%ACTIONS; // 10% mutation
        }
    }
}

__device__ int mn(int a,int b){return a<b?a:b;}

int main(){
    printf("=== MUD Arena with DCS ===\n");
    printf("%d agents, %d rooms, %d scripts, %d gens, %d turns/gen\n\n",NA,ROOMS,SCRIPTS,GENS,TURNS);
    
    srand(time(NULL));int seed=rand();
    int sb=(SCRIPTS*TURNS+BLK-1)/BLK,wb=(NA+BLK-1)/BLK;
    
    init_scripts<<<sb,BLK>>>(seed);
    init_world<<<wb,BLK>>>(seed);
    init_rooms<<<1,BLK>>>(seed+999);
    cudaDeviceSynchronize();
    
    // Run without DCS
    for(int g=0;g<GENS;g++){
        reset_world<<<wb,BLK>>>(seed+g*777);
        cudaDeviceSynchronize();
        sim_gen<<<wb,BLK>>>(g,0);
        cudaDeviceSynchronize();
        if(g<GENS-1){evolve<<<sb,BLK>>>();cudaDeviceSynchronize();}
    }
    int hf[SCRIPTS];cudaMemcpyFromSymbol(hf,fitness,sizeof(int)*SCRIPTS);
    int best_nodcs=0,best_idx=0;
    for(int i=0;i<SCRIPTS;i++)if(hf[i]>best_nodcs){best_nodcs=hf[i];best_idx=i;}
    int avg_nodcs=0;for(int i=0;i<SCRIPTS;i++)avg_nodcs+=hf[i];avg_nodcs/=SCRIPTS;
    
    // Run with DCS
    init_scripts<<<sb,BLK>>>(seed);
    init_world<<<wb,BLK>>>(seed);
    init_rooms<<<1,BLK>>>(seed+999);
    cudaDeviceSynchronize();
    
    for(int g=0;g<GENS;g++){
        reset_world<<<wb,BLK>>>(seed+g*777);
        cudaDeviceSynchronize();
        sim_gen<<<wb,BLK>>>(g,1);
        cudaDeviceSynchronize();
        if(g<GENS-1){evolve<<<sb,BLK>>>();cudaDeviceSynchronize();}
    }
    int hf2[SCRIPTS];cudaMemcpyFromSymbol(hf2,fitness,sizeof(int)*SCRIPTS);
    int best_dcs=0;for(int i=0;i<SCRIPTS;i++)if(hf2[i]>best_dcs)best_dcs=hf2[i];
    int avg_dcs=0;for(int i=0;i<SCRIPTS;i++)avg_dcs+=hf2[i];avg_dcs/=SCRIPTS;
    
    printf("=== Results ===\n");
    printf("NoDCS: best=%d avg=%d\n",best_nodcs,avg_nodcs);
    printf("DCS:   best=%d avg=%d\n",best_dcs,avg_dcs);
    printf("DCS lift: %.2fx (best), %.2fx (avg)\n",(float)best_dcs/best_nodcs,(float)avg_dcs/avg_nodcs);
    
    // Show top scripts
    printf("\nTop 5 NoDCS scripts:\n");
    int top[5]={0};for(int i=1;i<SCRIPTS;i++)for(int j=0;j<5;j++)if(hf[i]>hf[top[j]]){for(int l=4;l>j;l--)top[l]=top[l-1];top[j]=i;break;}
    for(int j=0;j<5;j++)printf("  #%d: score=%d\n",top[j],hf[top[j]]);
    
    return 0;
}
