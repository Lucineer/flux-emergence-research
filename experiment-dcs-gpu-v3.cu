/* experiment-dcs-gpu-v3.cu — DCS Protocol, simplified debugging
   Track per-agent contribution, separate solved state. */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NA 256
#define NTASKS 32
#define SUBS 3
#define MAXT 600
#define N_SKILLS 6

__device__ __host__ unsigned int lcg(unsigned int*s){*s=*s*1103515245u+12345u;return(*s>>16)&0x7fff;}
__device__ __host__ float lcgf(unsigned int*s){return(float)lcg(s)/32768.0f;}

typedef struct{float x,y;float skill[N_SKILLS];float energy;unsigned int rng;float fitness;int subs_solved;int tasks_completed;}Agent;

__global__ void init_agents(Agent*a,int n,int mode){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    a[i].rng=(unsigned int)(i*2654435761u+17);a[i].x=lcgf(&a[i].rng);a[i].y=lcgf(&a[i].rng);
    a[i].energy=1.0f;a[i].fitness=0;a[i].subs_solved=0;a[i].tasks_completed=0;
    if(mode==0){for(int s=0;s<N_SKILLS;s++)a[i].skill[s]=(s==(i%N_SKILLS))?(.7f+lcgf(&a[i].rng)*.3f):(.05f+lcgf(&a[i].rng)*.15f);}
    else{for(int s=0;s<N_SKILLS;s++)a[i].skill[s]=.25f+lcgf(&a[i].rng)*.2f;}
}

// Global task state
__device__ int g_task_sub_done[NTASKS*SUBS]; // 1=sub solved
__device__ int g_task_done[NTASKS];           // 1=task completed

__global__ void reset_tasks(int nt){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nt)return;
    g_task_done[i]=0;for(int s=0;s<SUBS;s++)g_task_sub_done[i*SUBS+s]=0;
}

__device__ float g_diff[NTASKS];
__device__ int g_req[NTASKS*SUBS];
__device__ float g_val[NTASKS*SUBS];

__global__ void setup_tasks(int nt,unsigned int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nt)return;
    unsigned int s=seed+(unsigned int)(i*9973);
    g_diff[i]=.3f+lcgf(&s)*.7f;
    for(int j=0;j<SUBS;j++){g_req[i*SUBS+j]=lcg(&s)%N_SKILLS;g_val[i*SUBS+j]=.5f+lcgf(&s)*.5f;}
}

__global__ void dcs_tick(Agent*a,int na,int nt,int tick_num){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    Agent*ag=&a[i];
    if(ag->energy<.15f){ag->energy+=.001f;return;} // rest
    
    // Pick a task (biased toward incomplete)
    int ti=-1;
    for(int tries=0;tries<3;tries++){
        int candidate=lcg(&ag->rng)%nt;
        if(!g_task_done[candidate]){ti=candidate;break;}
    }
    if(ti<0)return;
    
    // Find unsolved sub that matches our skills
    float best_skill=0;int best_si=-1;
    for(int s=0;s<SUBS;s++){
        if(g_task_sub_done[ti*SUBS+s])continue;
        float sk=ag->skill[g_req[ti*SUBS+s]];
        if(sk>best_skill){best_skill=sk;best_si=s;}
    }
    if(best_si<0||best_skill<.1f)return;
    
    // Attempt to solve
    ag->energy-=.02f;
    float solve_p=best_skill*(1.1f-g_diff[ti]*.5f);
    // DCS synergy: already-solved subs help
    int done_count=0;for(int s=0;s<SUBS;s++)if(g_task_sub_done[ti*SUBS+s])done_count++;
    solve_p*=(1+done_count*.15f);
    
    if(lcgf(&ag->rng)<solve_p){
        int idx=ti*SUBS+best_si;
        int old=atomicExch(&g_task_sub_done[idx],1);
        if(old==0){ // we were first
            ag->subs_solved++;
            // Check if all subs done
            int all_done=1;for(int s=0;s<SUBS;s++)if(!g_task_sub_done[ti*SUBS+s])all_done=0;
            if(all_done){
                int was_done=atomicExch(&g_task_done[ti],1);
                if(was_done==0){
                    float reward=0;for(int s=0;s<SUBS;s++)reward+=g_val[ti*SUBS+s];
                    ag->fitness+=reward*(1+best_skill*.3f);
                    ag->tasks_completed++;
                    ag->energy=fminf(1,ag->energy+.1f);
                }
            }
        }
    }
    ag->energy*=.9998f;
}

__global__ void indiv_tick(Agent*a,int na,int nt){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    Agent*ag=&a[i];
    if(ag->energy<.15f){ag->energy+=.001f;return;}
    
    int ti=lcg(&ag->rng)%nt;
    if(g_task_done[ti])return;
    
    float best_skill=0;int best_si=-1;
    for(int s=0;s<SUBS;s++){
        if(g_task_sub_done[ti*SUBS+s])continue;
        float sk=ag->skill[g_req[ti*SUBS+s]];
        if(sk>best_skill){best_skill=sk;best_si=s;}
    }
    if(best_si<0||best_skill<.1f)return;
    
    ag->energy-=.04f;
    // Individual: no DCS synergy bonus, harder
    float solve_p=best_skill*best_skill*(1.1f-g_diff[ti]*.5f)*.4f;
    
    if(lcgf(&ag->rng)<solve_p){
        int idx=ti*SUBS+best_si;
        int old=atomicExch(&g_task_sub_done[idx],1);
        if(old==0){
            ag->subs_solved++;
            int all_done=1;for(int s=0;s<SUBS;s++)if(!g_task_sub_done[ti*SUBS+s])all_done=0;
            if(all_done){
                int was_done=atomicExch(&g_task_done[ti],1);
                if(was_done==0){
                    float reward=0;for(int s=0;s<SUBS;s++)reward+=g_val[ti*SUBS+s];
                    ag->fitness+=reward*best_skill;
                    ag->tasks_completed++;
                }
            }
        }
    }
    ag->energy*=.9998f;
}

__global__ void respawn_tasks(int nt,unsigned int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nt)return;
    if(!g_task_done[i])return;
    unsigned int s=seed+(unsigned int)(i*7727);
    g_diff[i]=.3f+lcgf(&s)*.7f;
    for(int j=0;j<SUBS;j++){g_req[i*SUBS+j]=lcg(&s)%N_SKILLS;g_val[i*SUBS+j]=.5f+lcgf(&s)*.5f;g_task_sub_done[i*SUBS+j]=0;}
    g_task_done[i]=0;
}

__global__ void sum_results(Agent*a,int na,float*total_fit,int*total_subs,int*total_tasks,int*active){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=na)return;
    atomicAdd(total_fit,a[i].fitness);
    atomicAdd(total_subs,a[i].subs_solved);
    atomicAdd(total_tasks,a[i].tasks_completed);
    if(a[i].energy>.1f)atomicAdd(active,1);
}

int main(){
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Experiment: DCS Protocol v3\n");
    printf("  %d agents, %d tasks×%d subs, 6 skill types\n",NA,NTASKS,SUBS);
    printf("═══════════════════════════════════════════════════════\n\n");
    Agent*da;cudaMalloc(&da,NA*sizeof(Agent));
    float*d_fit;int*d_subs,*d_tasks,*d_active;
    cudaMalloc(&d_fit,sizeof(float));cudaMalloc(&d_subs,sizeof(int));cudaMalloc(&d_tasks,sizeof(int));cudaMalloc(&d_active,sizeof(int));
    Agent*ha=(Agent*)malloc(NA*sizeof(Agent));
    int blk=(NA+255)/256,tblk=(NTASKS+255)/256;
    
    const char*names[]={"DCS Specialists","DCS Generalists","Individual Specialists","Individual Generalists"};
    float results[4]={0};
    int total_subs_arr[4]={0},total_tasks_arr[4]={0};
    
    for(int mode=0;mode<4;mode++){
        int agent_mode=mode%2,proto=mode/2;
        for(int run=0;run<5;run++){
            unsigned int seed=(unsigned int)(run*2654435761u+mode*99991);
            init_agents<<<blk,256>>>(da,NA,agent_mode);reset_tasks<<<tblk,256>>>(NTASKS);setup_tasks<<<tblk,256>>>(NTASKS,seed);cudaDeviceSynchronize();
            for(int t=0;t<MAXT;t++){
                if(proto==0)dcs_tick<<<blk,256>>>(da,NA,NTASKS,t);
                else indiv_tick<<<blk,256>>>(da,NA,NTASKS);
                if(t%25==0)respawn_tasks<<<tblk,256>>>(NTASKS,seed+t);
                cudaDeviceSynchronize();
            }
            cudaMemcpy(ha,da,NA*sizeof(Agent),cudaMemcpyDeviceToHost);
            float f=0;int ts=0,tt=0;
            for(int i=0;i<NA;i++){f+=ha[i].fitness;ts+=ha[i].subs_solved;tt+=ha[i].tasks_completed;}
            results[mode]+=f;total_subs_arr[mode]+=ts;total_tasks_arr[mode]+=tt;
        }
        results[mode]/=5;total_subs_arr[mode]/=5;total_tasks_arr[mode]/=5;
        printf("  %-25s: fit=%.1f subs=%d tasks=%d\n",names[mode],results[mode],total_subs_arr[mode],total_tasks_arr[mode]);
    }
    printf("\n─── Analysis ───\n");
    float ds=results[0],dg=results[1],is=results[2],ig=results[3];
    printf("  DCS spec vs indiv spec:  %.2fx (%+.0f%%)\n",(is>.01)?ds/is:1,((is>.01)?(ds/is-1)*100:0));
    printf("  DCS gen vs indiv gen:    %.2fx (%+.0f%%)\n",(ig>.01)?dg/ig:1,((ig>.01)?(dg/ig-1)*100:0));
    printf("  Spec vs Gen (DCS):       %.2fx\n",(dg>.01)?ds/dg:1);
    printf("  Spec vs Gen (Indiv):     %.2fx\n",(ig>.01)?is/ig:1);
    if(ds>is&&dg>ig)printf("\n  → DCS PROTOCOL WINS for both specialists and generalists\n");
    else if(ds<is)printf("\n  → Individual beats DCS for specialists (counterintuitive)\n");
    printf("═══════════════════════════════════════════════════════\n");
    cudaFree(da);cudaFree(d_fit);cudaFree(d_subs);cudaFree(d_tasks);cudaFree(d_active);free(ha);return 0;
}
