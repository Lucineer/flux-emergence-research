#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 512
#define STEPS 3000
#define W 256
#define BLK 128
#define NMODES 3
#define TRIALS 5

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(int *alive, int steps, int n, int w,
    int mode, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng)*w, y = cr(&rng)*w;
    float base_angle = tid * 2.39996f;
    float dir[8];for(int i=0;i<8;i++)dir[i]=base_angle+i*0.785f;
    int my_alive = 1;
    // Predator
    unsigned int prng = seed + 999999;
    float px = cr(&prng)*w, py = cr(&prng)*w;
    float kill_r2 = 100.0f;
    for(int t=0;t<steps&&my_alive;t++){
        // Predator moves randomly at speed 3
        float pa = cr(&prng)*6.2832f;
        px=fmodf(px+cosf(pa)*3.0f+w,w);
        py=fmodf(py+sinf(pa)*3.0f+w,w);
        // Agent moves based on mode
        float dx=0,dy=0;
        if(mode==0){ // script
            int p=t%8;dx=cosf(dir[p])*2.0f;dy=sinf(dir[p])*2.0f;
        } else if(mode==1){ // random walk
            float a=cr(&rng)*6.2832f;dx=cosf(a)*2.0f;dy=sinf(a)*2.0f;
        }
        // mode==2: stationary
        float dist=sqrtf(dx*dx+dy*dy);
        if(mode!=2){x=fmodf(x+dx+w,w);y=fmodf(y+dy+w,w);}
        // Check predator collision
        float pdx=px-x,pdy=py-y;
        if(pdx>w/2)pdx-=w;if(pdx<-w/2)pdx+=w;
        if(pdy>w/2)pdy-=w;if(pdy<-w/2)pdy+=w;
        if(pdx*pdx+pdy*pdy<kill_r2) my_alive=0;
    }
    alive[tid]=my_alive;
}

int main(){
    const char* nm[]={"Script","RandomWalk","Stationary"};
    printf("=== PREDATOR vs SCRIPTS: Law 251 ===\nN=%d Steps=%d World=%d Trials=%d\n\n",N,STEPS,W,TRIALS);
    int *d_a;cudaMalloc(&d_a,N*sizeof(int));
    int blk=(N+BLK-1)/BLK;
    for(int m=0;m<NMODES;m++){
        float surv=0;
        for(int tr=0;tr<TRIALS;tr++){
            simulate<<<blk,BLK>>>(d_a,STEPS,N,W,m,(unsigned int)(42+tr*1111+m*111));
            cudaDeviceSynchronize();
            int ha[N];cudaMemcpy(ha,d_a,N*sizeof(int),cudaMemcpyDeviceToHost);
            int ac=0;for(int i=0;i<N;i++)ac+=ha[i];
            surv+=ac/(float)N*100;
        }
        surv/=TRIALS;
        printf("%-16s: survival=%.1f%%\n",nm[m],surv);
    }
    printf("\n>> Law 251: Do scripted agents survive predators better?\n");
    cudaFree(d_a);
    return 0;
}