#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
__device__ int mx(int a,int b){return a>b?a:b;}

#define NA 2048
#define ROOMS 256
#define BLK 128
#define GENS 200
#define TURNS 200
#define SCRIPTS 16
#define TEAMS 2

__device__ int scripts[SCRIPTS*TEAMS][TURNS];
__device__ int fitness[SCRIPTS*TEAMS];
__device__ int room_gold[ROOMS];
__device__ int a_hp[NA],a_gold[NA],a_room[NA],a_script[NA],a_alive[NA],a_team[NA];
__device__ int aseed[NA];
__device__ int dcs_room[TEAMS],dcs_gold[TEAMS],dcs_v[TEAMS];

__device__ int rn(int*s){*s=(*s*1103515245+12345)&0x7fffffff;return*s;}

__global__ void init_scripts(int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=SCRIPTS*TEAMS*TURNS)return;
    scripts[i/TURNS][i%TURNS]=rn(&seed)%(4); // 4 actions: idle/attack/trade/flee
}

__global__ void init_world(int seed,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NA)return;
    aseed[i]=seed+i*137;a_hp[i]=100;a_gold[i]=10;a_room[i]=rn(&aseed[i])%ROOMS;
    a_team[i]=i<NA/2?0:1;
    a_script[i]=a_team[i]*SCRIPTS+(rn(&aseed[i])%SCRIPTS);
    a_alive[i]=1;
    if(i<ROOMS){room_gold[i]=rn(&seed+i)%50;}
    if(i<TEAMS)dcs_v[i]=0;
}

__global__ void sim_gen(int gen,int use_dcs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=NA)return;
    if(i==0)for(int r=0;r<ROOMS;r++)room_gold[r]=rn(&aseed[0]+r)%50;
    
    if(!a_alive[i])return;
    int act=scripts[a_script[i]][gen%TURNS];
    int rm=a_room[i];
    int team=a_team[i];
    
    // DCS: move to team's best room
    if(use_dcs&&dcs_v[team]&&dcs_room[team]!=rm&&dcs_gold[team]>room_gold[rm]+3){
        a_room[i]=dcs_room[team];rm=dcs_room[team];act=2; // pickup
    }
    
    if(act==0){/* idle */}
    else if(act==1){/* attack room */
        if(room_gold[rm]>0){room_gold[rm]=mx(room_gold[rm]-2,0);a_gold[i]+=3;}
        else{a_hp[i]-=3;if(a_hp[i]<=0)a_alive[i]=0;}
    }else if(act==2){/* trade */
        if(a_gold[i]>5){a_gold[i]-=5;a_hp[i]+=10;}
    }else{/* flee */
        a_room[i]=(a_room[i]+rn(&aseed[i])%ROOMS)%ROOMS;
    }
    
    // Update DCS on room discovery
    if(room_gold[rm]>dcs_gold[team]){
        dcs_room[team]=rm;dcs_gold[team]=room_gold[rm];dcs_v[team]=1;
    }
}

__global__ void score(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=SCRIPTS*TEAMS)return;
    fitness[i]=0;
    int team=i/SCRIPTS,script=i%SCRIPTS;
    for(int j=0;j<NA;j++){
        if(a_team[j]==team&&a_script[j]==team*SCRIPTS+script){
            fitness[i]+=a_alive[j]?a_gold[j]+50:a_gold[j];
        }
    }
}

__global__ void evolve(){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=SCRIPTS*TEAMS)return;
    int team=i/SCRIPTS;int base=team*SCRIPTS;
    int best[4],bv[4];
    for(int j=0;j<4;j++){best[j]=base+j;bv[j]=fitness[base+j];}
    for(int j=4;j<SCRIPTS;j++){
        int idx=base+j;
        for(int k=0;k<4;k++){if(fitness[idx]>bv[k]){
            for(int l=3;l>k;l--){bv[l]=bv[l-1];best[l]=best[l-1];}
            bv[k]=fitness[idx];best[k]=idx;break;}}
    }
    if(i-base<4){
        for(int j=0;j<TURNS;j++)scripts[i][j]=scripts[best[i-base]][j];
    }else{
        int parent=best[rn(&aseed[i])%4];
        for(int j=0;j<TURNS;j++){
            scripts[i][j]=scripts[parent][j];
            if(rn(&aseed[i])%100<10)scripts[i][j]=rn(&aseed[i])%4;
        }
    }
}


int main(){
    printf("=== MUD Team Competition ===\n");
    printf("%d agents (%d/team), %d rooms, %d scripts/team\n",NA,NA/TEAMS,ROOMS,SCRIPTS);
    
    srand(time(NULL));int seed=rand();
    int sb=(SCRIPTS*TEAMS*TURNS+BLK-1)/BLK,wb=(NA+BLK-1)/BLK;
    
    float red_nodcs=0,blue_nodcs=0,red_dcs=0,blue_dcs=0;
    
    // Run without DCS
    for(int trial=0;trial<3;trial++){
        init_scripts<<<sb,BLK>>>(seed+trial);
        for(int g=0;g<GENS;g++){
            init_world<<<wb,BLK>>>(seed+trial*777+g,0);
            cudaDeviceSynchronize();
            sim_gen<<<wb,BLK>>>(g,0);
            if(g%50==0)cudaDeviceSynchronize();
            if(g<GENS-1){score<<<sb,BLK>>>();evolve<<<sb,BLK>>>();cudaDeviceSynchronize();}
        }
        score<<<sb,BLK>>>();cudaDeviceSynchronize();
        int hf[SCRIPTS*TEAMS];cudaMemcpyFromSymbol(hf,fitness,sizeof(int)*SCRIPTS*TEAMS);
        for(int t=0;t<SCRIPTS;t++){red_nodcs+=hf[t];blue_nodcs+=hf[SCRIPTS+t];}
    }
    red_nodcs/=3*SCRIPTS;blue_nodcs/=3*SCRIPTS;
    
    // Run with DCS
    for(int trial=0;trial<3;trial++){
        init_scripts<<<sb,BLK>>>(seed+trial);
        for(int g=0;g<GENS;g++){
            init_world<<<wb,BLK>>>(seed+trial*777+g,1);
            cudaDeviceSynchronize();
            sim_gen<<<wb,BLK>>>(g,1);
            if(g%50==0)cudaDeviceSynchronize();
            if(g<GENS-1){score<<<sb,BLK>>>();evolve<<<sb,BLK>>>();cudaDeviceSynchronize();}
        }
        score<<<sb,BLK>>>();cudaDeviceSynchronize();
        int hf[SCRIPTS*TEAMS];cudaMemcpyFromSymbol(hf,fitness,sizeof(int)*SCRIPTS*TEAMS);
        for(int t=0;t<SCRIPTS;t++){red_dcs+=hf[t];blue_dcs+=hf[SCRIPTS+t];}
    }
    red_dcs/=3*SCRIPTS;blue_dcs/=3*SCRIPTS;
    
    printf("\n=== Results (avg fitness/script) ===\n");
    printf("          | NoDCS | DCS   | Lift\n");
    printf("Red team  | %5.0f | %5.0f | %.2fx\n",red_nodcs,red_dcs,red_dcs/red_nodcs);
    printf("Blue team | %5.0f | %5.0f | %.2fx\n",blue_nodcs,blue_dcs,blue_dcs/blue_nodcs);
    
    return 0;
}
