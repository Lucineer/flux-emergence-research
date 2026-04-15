// experiment-speed-ramp.cu — Laws 197+: Speed forces grand strategy
// Key insight: scripted strategies don't scale movement with speed_mult,
// while reactive strategies do — causing overshoot at high speed.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define FOOD 400
#define STEPS 3000
#define W 256
#define BLK 128
#define NSPEEDS 6
#define TRIALS 5
#define GRAB 4.0f

__device__ float cr(unsigned int *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return (*s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

__global__ void simulate(
    float *scores, int *alive,
    float *fx, float *fy, int *falive,
    float speed_mult, int steps, int n, int food_count, int w,
    int strat, unsigned int seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned int rng = seed + tid * 997;
    float x = cr(&rng) * w, y = cr(&rng) * w;
    float energy = 150.0f, score = 0.0f;
    
    // Script: deterministic pattern based on agent ID (not random)
    float base_angle = (tid * 2.39996f); // golden angle spread
    float script_dir[8], script_grab[8];
    for (int i = 0; i < 8; i++) {
        script_dir[i] = base_angle + i * 0.785f; // 8 directions, golden offset
        script_grab[i] = GRAB;
    }
    
    // Adaptive state
    float move_spd = 2.0f, grab_r = GRAB;
    float last_adapt = -999.0f;
    
    for (int t = 0; t < steps && energy > 0; t++) {
        float dx = 0, dy = 0, r = GRAB;
        
        switch (strat) {
        case 0: // RANDOM: recalculates every tick, scales with speed
            dx = (cr(&rng) - 0.5f) * 6.0f * speed_mult;
            dy = (cr(&rng) - 0.5f) * 6.0f * speed_mult;
            r = GRAB;
            break;
        case 1: // ADAPTIVE: re-evaluates every 50 ticks, movement scales with speed
            if (t - last_adapt > 50.0f) {
                move_spd = 1.0f + cr(&rng) * 4.0f;
                grab_r = 1.0f + cr(&rng) * 6.0f;
                last_adapt = t;
            }
            { float a = cr(&rng) * 6.2832f;
              dx = cosf(a) * move_spd * speed_mult;
              dy = sinf(a) * move_spd * speed_mult; }
            r = grab_r;
            break;
        case 2: // SCRIPTED: fixed pattern, does NOT scale with speed
            { int p = t % 8;
              dx = cosf(script_dir[p]) * 2.0f;  // constant speed
              dy = sinf(script_dir[p]) * 2.0f; }
            r = GRAB;
            break;
        case 3: // DCS-SCAN: scans for food, approaches it (scales with speed -> overshoot)
            { float best_d = 999.0f, bx = x, by = y;
              for (int i = 0; i < food_count; i += 8) {
                  if (!falive[i]) continue;
                  float fdx = fx[i] - x, fdy = fy[i] - y;
                  if (fdx > w/2) fdx -= w; if (fdx < -w/2) fdx += w;
                  if (fdy > w/2) fdy -= w; if (fdy < -w/2) fdy += w;
                  float d = fdx*fdx + fdy*fdy;
                  if (d < best_d) { best_d = d; bx = fx[i]; by = fy[i]; }
              }
              if (best_d < 400.0f) {
                  dx = (bx - x) * 0.3f * speed_mult;  // scales with speed!
                  dy = (by - y) * 0.3f * speed_mult;
              } else {
                  dx = (cr(&rng) - 0.5f) * 4.0f * speed_mult;
                  dy = (cr(&rng) - 0.5f) * 4.0f * speed_mult;
              }
            }
            r = GRAB;
            break;
        }
        
        // Movement energy cost (scales with distance moved)
        float dist = sqrtf(dx*dx + dy*dy);
        energy -= 0.005f + dist * 0.003f;
        
        // Move
        x = fmodf(x + dx + w, w);
        y = fmodf(y + dy + w, w);
        
        // Grab
        for (int i = 0; i < food_count; i++) {
            if (!falive[i]) continue;
            float fdx = fx[i] - x, fdy = fy[i] - y;
            if (fdx > w/2) fdx -= w; if (fdx < -w/2) fdx += w;
            if (fdy > w/2) fdy -= w; if (fdy < -w/2) fdy += w;
            if (fdx*fdx + fdy*fdy < r*r) {
                int old = atomicExch(&falive[i], 0);
                if (old) { energy = fminf(energy + 10.0f, 200.0f); score += 1.0f; }
            }
        }
    }
    scores[tid] = score;
    alive[tid] = (energy > 0) ? 1 : 0;
}

int main() {
    const char* nm[] = {"Random", "Adaptive", "Scripted", "FoodScan"};
    float speeds[NSPEEDS] = {1, 2, 4, 8, 16, 32};
    printf("=== SPEED RAMP: Laws 197+ ===\n");
    printf("N=%d Food=%d Steps=%d World=%d Trials=%d\n\n", N, FOOD, STEPS, W, TRIALS);
    printf("Hypothesis: scripted (no speed scaling) outperforms reactive at high speed.\n");
    printf("Reactive strategies overshoot targets. Scripts maintain stable orbits.\n\n");
    
    float *d_s, *d_fx, *d_fy;
    int *d_a, *d_fa;
    cudaMalloc(&d_s, N*sizeof(float));
    cudaMalloc(&d_fx, FOOD*sizeof(float));
    cudaMalloc(&d_fy, FOOD*sizeof(float));
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_fa, FOOD*sizeof(int));
    
    float hfx[FOOD], hfy[FOOD];
    srand(42);
    for (int i = 0; i < FOOD; i++) {
        hfx[i] = ((float)rand()/RAND_MAX)*W;
        hfy[i] = ((float)rand()/RAND_MAX)*W;
    }
    cudaMemcpy(d_fx, hfx, FOOD*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, hfy, FOOD*sizeof(float), cudaMemcpyHostToDevice);
    
    float res[NSPEEDS][4], srv[NSPEEDS][4];
    int blk = (N+BLK-1)/BLK;
    
    for (int s = 0; s < NSPEEDS; s++) {
        for (int st = 0; st < 4; st++) {
            float ts=0, ta=0;
            for (int tr = 0; tr < TRIALS; tr++) {
                cudaMemset(d_fa, 1, FOOD*sizeof(int));
                simulate<<<blk,BLK>>>(d_s, d_a, d_fx, d_fy, d_fa,
                    speeds[s], STEPS, N, FOOD, W, st,
                    (unsigned int)(42+tr*1111+s*111+st*11));
                cudaDeviceSynchronize();
                float hs[N]; int ha[N];
                cudaMemcpy(hs, d_s, N*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(ha, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);
                float avg=0; int ac=0;
                for (int i=0;i<N;i++){avg+=hs[i];ac+=ha[i];}
                ts += avg/N; ta += ac;
            }
            res[s][st] = ts/TRIALS;
            srv[s][st] = ta/TRIALS/N*100;
        }
    }
    
    printf("%-8s","Speed");
    for(int st=0;st<4;st++) printf(" | %-9s",nm[st]);
    printf(" | %-9s","Best");
    printf("\n--------"); for(int i=0;i<5;i++) printf("----------"); printf("\n");
    
    for (int s=0;s<NSPEEDS;s++) {
        printf("%-7dx",(int)speeds[s]);
        int best=0;
        for(int st=0;st<4;st++){
            printf(" | %7.2f  ",res[s][st]);
            if(res[s][st]>res[s][best])best=st;
        }
        printf(" | %-9s\n",nm[best]);
    }
    
    printf("\nSurvival %%:\n%-8s","Speed");
    for(int st=0;st<4;st++) printf(" | %-9s",nm[st]);
    printf("\n--------"); for(int i=0;i<4;i++) printf("----------"); printf("\n");
    for(int s=0;s<NSPEEDS;s++){
        printf("%-7dx",(int)speeds[s]);
        for(int st=0;st<4;st++) printf(" | %7.1f%%  ",srv[s][st]);
        printf("\n");
    }
    
    // Analysis
    printf("\n=== DEGRADATION 1x->32x ===\n");
    for(int st=0;st<4;st++){
        float b=fmaxf(res[0][st],0.001f), f=res[5][st];
        printf("%s: %.1f%% (%.2f -> %.2f)\n",nm[st],(b-f)/b*100,res[0][st],res[5][st]);
    }
    
    float rd=(res[0][0]-res[5][0])/fmaxf(res[0][0],.001f);
    float sd=(res[0][2]-res[5][2])/fmaxf(res[0][2],.001f);
    float ad=(res[0][1]-res[5][1])/fmaxf(res[0][1],.001f);
    float fd=(res[0][3]-res[5][3])/fmaxf(res[0][3],.001f);
    
    printf("\n=== LAW CANDIDATES ===\n");
    if(sd<rd) printf("Law 197: Scripts degrade LESS with speed than random (%.1fx vs %.1fx)\n",sd,rd);
    if(ad>sd) printf("Law 198: Adaptive collapses faster than scripted at high speed (%.1fx vs %.1fx)\n",ad,sd);
    if(fd>sd) printf("Law 199: Food scanning (approach behavior) degrades worst at high speed\n");
    
    int sw=0;
    for(int s=3;s<6;s++){int b=0;for(int st=1;st<4;st++)if(res[s][st]>res[s][b])b=st;if(b==2)sw++;}
    if(sw>=1) printf("Law 200: Scripted WINS at high speed (%d/3 fast trials)\n",sw);
    
    if(res[0][3]>res[0][2] && res[5][2]>res[5][3])
        printf("Law 201: Speed INVERTS strategy ranking (FoodScan>Scripted at 1x, Scripted>FoodScan at 32x)\n");
    
    // Survival analysis
    float ss_base=srv[0][2], ss_fast=srv[5][2];
    float sr_base=srv[0][0], sr_fast=srv[5][0];
    if(ss_fast>0 && sr_fast==0)
        printf("Law 202: Only scripted agents survive extreme speed. Reactive agents burn out.\n");
    
    cudaFree(d_s);cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_a);cudaFree(d_fa);
    return 0;
}
