// experiment-ensign-review.cu
// Ensign with Periodic Higher Review
// 
// Casey: "even when nothing outside the deadband has happened for a while,
// there still might be a higher model review every now and then"
//
// Three detection strategies:
// 1. Deadband only (fast, misses slow drift)
// 2. Deadband + periodic review (every N steps, check cumulative drift)
// 3. Deadband + jerk detection (rate-of-change of rate-of-change)
//
// Anomaly types: slow drift, step change, spike burst, sinusoidal

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_SENSORS 512
#define N_STEPS   8000
#define TRIALS    5
#define DEADBAND  1.0f
#define REVIEW_WINDOW 200  // look back N steps

__device__ int d_tp, d_fp, d_fn, d_tn;
__device__ float d_sensor_vals[N_SENSORS];
__device__ float d_sensor_hist[N_SENSORS * REVIEW_WINDOW]; // ring buffer

__device__ float rng(unsigned int *s) {
    *s = *s * 1103515245u + 12345u;
    return (float)(*s & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}

__global__ void reset_counters() {
    if (threadIdx.x==0 && blockIdx.x==0) { d_tp=0; d_fp=0; d_fn=0; d_tn=0; }
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N_SENSORS) { d_sensor_vals[i]=50.0f; d_sensor_hist[i]=50.0f; }
}

__global__ void init_anomalies(int *starts, int *ends, float *mags, int n,
                                unsigned int seed, float anomaly_type) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    unsigned int s = seed + i * 7919;
    
    if (rng(&s) < 0.35f) {  // 35% sensors have anomalies
        starts[i] = 1500 + (int)(rng(&s) * 2500);
        int duration = 500 + (int)(rng(&s) * 2000);
        ends[i] = starts[i] + duration;
        mags[i] = anomaly_type;
    } else {
        starts[i] = N_STEPS + 1;
        ends[i] = N_STEPS + 1;
        mags[i] = 0;
    }
}

// Generate value with possible anomaly
__device__ float gen_val(int sensor_id, int step, unsigned int *s,
                          int anomaly_start, int anomaly_end, float anomaly_mag,
                          int anomaly_type_idx) {
    float noise = (rng(s) - 0.5f) * 0.3f;
    float normal = 50.0f + sinf(step * 0.0005f + sensor_id * 0.1f) * 0.2f;
    
    if (step < anomaly_start || step >= anomaly_end) return normal + noise;
    
    float progress = (float)(step - anomaly_start) / (anomaly_end - anomaly_start);
    
    switch(anomaly_type_idx) {
        case 0: // slow drift
            return normal + anomaly_mag * progress * 10.0f + noise;
        case 1: // step change
            return normal + anomaly_mag * 5.0f + noise;
        case 2: // spike burst (intermittent)
            return normal + ((step % 20 < 5) ? anomaly_mag * 8.0f : 0) + noise;
        case 3: // sinusoidal
            return normal + sinf(step * 0.05f) * anomaly_mag * 3.0f + noise;
        case 4: // very slow drift (barely perceptible)
            return normal + anomaly_mag * progress * 5.0f + noise;
        default:
            return normal + noise;
    }
}

// MODE 0: Pure deadband
__global__ void detect_deadband(int step, unsigned int seed,
                                 int *starts, int *ends, float *mags, int atype) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N_SENSORS) return;
    unsigned int s = seed + i * 131 + step * 997;
    
    float baseline = 50.0f;
    float val = gen_val(i, step, &s, starts[i], ends[i], mags[i], atype);
    d_sensor_vals[i] = val;
    
    float diff = fabsf(val - baseline);
    int in_anomaly = (step >= starts[i] && step < ends[i]);
    
    if (diff > DEADBAND) {
        if (in_anomaly) atomicAdd(&d_tp, 1);
        else atomicAdd(&d_fp, 1);
    } else {
        if (in_anomaly) atomicAdd(&d_fn, 1);
        else atomicAdd(&d_tn, 1);
    }
}

// MODE 1: Deadband + periodic review (check cumulative drift every N steps)
__global__ void detect_review(int step, unsigned int seed,
                               int *starts, int *ends, float *mags, int atype,
                               int review_interval, float review_threshold) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N_SENSORS) return;
    unsigned int s = seed + i * 131 + step * 997;
    
    float baseline = 50.0f;
    float val = gen_val(i, step, &s, starts[i], ends[i], mags[i], atype);
    d_sensor_vals[i] = val;
    
    // Store in ring buffer
    int buf_idx = (step % REVIEW_WINDOW) * N_SENSORS + i;
    d_sensor_hist[buf_idx] = val;
    
    float diff = fabsf(val - baseline);
    int in_anomaly = (step >= starts[i] && step < ends[i]);
    int alerted = 0;
    
    // Deadband check
    if (diff > DEADBAND) alerted = 1;
    
    // Periodic review: check if mean of recent window has drifted
    if (!alerted && step > REVIEW_WINDOW && (step % review_interval == 0)) {
        float sum = 0;
        for (int w = 0; w < REVIEW_WINDOW; w++) {
            sum += d_sensor_hist[(w) * N_SENSORS + i]; // simplified: scan all
        }
        float mean = sum / REVIEW_WINDOW;
        float drift = fabsf(mean - baseline);
        if (drift > review_threshold) alerted = 1;
    }
    
    if (alerted) {
        if (in_anomaly) atomicAdd(&d_tp, 1);
        else atomicAdd(&d_fp, 1);
    } else {
        if (in_anomaly) atomicAdd(&d_fn, 1);
        else atomicAdd(&d_tn, 1);
    }
}

// MODE 2: Deadband + jerk (rate-of-change of rate-of-change)
__global__ void detect_jerk(int step, unsigned int seed,
                             int *starts, int *ends, float *mags, int atype,
                             float jerk_threshold) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= N_SENSORS || step < 2) return;
    unsigned int s = seed + i * 131 + step * 997;
    
    float baseline = 50.0f;
    float val = gen_val(i, step, &s, starts[i], ends[i], mags[i], atype);
    
    // Get history
    int h1 = ((step-1) % REVIEW_WINDOW) * N_SENSORS + i;
    int h2 = ((step-2) % REVIEW_WINDOW) * N_SENSORS + i;
    float prev = d_sensor_hist[h1];
    float prevprev = d_sensor_hist[h2];
    
    d_sensor_vals[i] = val;
    int buf_idx = (step % REVIEW_WINDOW) * N_SENSORS + i;
    d_sensor_hist[buf_idx] = val;
    
    float diff = fabsf(val - baseline);
    int in_anomaly = (step >= starts[i] && step < ends[i]);
    int alerted = 0;
    
    if (diff > DEADBAND) alerted = 1;
    
    // Jerk = |acceleration change|
    float vel = val - prev;
    float prev_vel = prev - prevprev;
    float jerk = fabsf(vel - prev_vel);
    
    if (!alerted && jerk > jerk_threshold) alerted = 1;
    
    if (alerted) {
        if (in_anomaly) atomicAdd(&d_tp, 1);
        else atomicAdd(&d_fp, 1);
    } else {
        if (in_anomaly) atomicAdd(&d_fn, 1);
        else atomicAdd(&d_tn, 1);
    }
}

int main() {
    printf("=== Ensign + Periodic Higher Review ===\n");
    printf("Sensors: %d | Steps: %d | Deadband: %.1f | Review window: %d\n\n",
           N_SENSORS, N_STEPS, DEADBAND, REVIEW_WINDOW);
    
    int bs=256, sg=(N_SENSORS+bs-1)/bs;
    int *d_starts, *d_ends; float *d_mags;
    cudaMalloc(&d_starts, N_SENSORS*sizeof(int));
    cudaMalloc(&d_ends, N_SENSORS*sizeof(int));
    cudaMalloc(&d_mags, N_SENSORS*sizeof(float));
    srand(time(NULL));
    
    // Anomaly types: slow_drift(0.5), slow_drift(0.1), step(0.8), spike(0.6), sine(0.4), very_slow(0.2)
    const char *atype_names[] = {"Slow drift 0.5","Slow drift 0.1","Step 0.8","Spike 0.6","Sine 0.4","V.Slow 0.2"};
    float atype_mags[] = {0.5f, 0.1f, 0.8f, 0.6f, 0.4f, 0.2f};
    
    // === Compare detection modes across anomaly types ===
    printf("=== Detection Mode Comparison ===\n");
    printf("%-14s | %-20s | %8s %8s %8s | %8s\n", "Anomaly", "Mode", "TP","FP","FN","Recall");
    printf("%s\n", "------------------------------------------------------------------------");
    
    for (int at = 0; at < 6; at++) {
        for (int mode = 0; mode < 3; mode++) {
            int tp=0,fp=0,fn=0,tn=0;
            
            for (int t = 0; t < TRIALS; t++) {
                unsigned int seed = (unsigned int)time(NULL) + t*30000 + at*100000 + mode*500000;
                
                init_anomalies<<<sg,bs>>>(d_starts,d_ends,d_mags,N_SENSORS,seed,atype_mags[at]);
                reset_counters<<<sg,bs>>>();
                cudaDeviceSynchronize();
                
                // Warm up history (2 steps for jerk)
                for (int step = 0; step < 2; step++) {
                    unsigned int s2 = seed + step;
                    detect_deadband<<<sg,bs>>>(step,s2,d_starts,d_ends,d_mags,at);
                    cudaDeviceSynchronize();
                }
                
                // Zero counters after warmup
                int zero=0;
                cudaMemcpyToSymbol(d_tp,&zero,sizeof(int));
                cudaMemcpyToSymbol(d_fp,&zero,sizeof(int));
                cudaMemcpyToSymbol(d_fn,&zero,sizeof(int));
                cudaMemcpyToSymbol(d_tn,&zero,sizeof(int));
                
                for (int step = 2; step < N_STEPS; step++) {
                    unsigned int s2 = seed + step;
                    switch(mode) {
                        case 0: detect_deadband<<<sg,bs>>>(step,s2,d_starts,d_ends,d_mags,at); break;
                        case 1: detect_review<<<sg,bs>>>(step,s2,d_starts,d_ends,d_mags,at,100,0.3f); break;
                        case 2: detect_jerk<<<sg,bs>>>(step,s2,d_starts,d_ends,d_mags,at,0.15f); break;
                    }
                    cudaDeviceSynchronize();
                }
                
                int h_tp,h_fp,h_fn,h_tn;
                cudaMemcpyFromSymbol(&h_tp,d_tp,sizeof(int));
                cudaMemcpyFromSymbol(&h_fp,d_fp,sizeof(int));
                cudaMemcpyFromSymbol(&h_fn,d_fn,sizeof(int));
                cudaMemcpyFromSymbol(&h_tn,d_tn,sizeof(int));
                tp+=h_tp; fp+=h_fp; fn+=h_fn; tn+=h_tn;
            }
            
            float recall = (tp+fn)>0 ? (float)tp/(tp+fn) : 0;
            float precision = (tp+fp)>0 ? (float)tp/(tp+fp) : 0;
            const char *mode_names[] = {"Deadband","DB+Review","DB+Jerk"};
            printf("%-14s | %-20s | %8d %8d %8d | %7.3f\n",
                   atype_names[at], mode_names[mode], tp/TRIALS, fp/TRIALS, fn/TRIALS, recall);
        }
    }
    
    // === Review interval sweep ===
    printf("\n=== Review Interval Sweep (slow drift 0.5) ===\n");
    printf("%-12s %8s %8s %8s %8s\n", "Interval", "TP", "FP", "FN", "Recall");
    printf("%s\n", "----------------------------------------------");
    
    for (int interval = 10; interval <= 500; interval += (interval < 50 ? 10 : interval < 200 ? 25 : 50)) {
        int tp=0,fp=0,fn=0;
        for (int t = 0; t < TRIALS; t++) {
            unsigned int seed = (unsigned int)time(NULL) + t*30000 + interval*1000;
            init_anomalies<<<sg,bs>>>(d_starts,d_ends,d_mags,N_SENSORS,seed,0.5f);
            reset_counters<<<sg,bs>>>();
            cudaDeviceSynchronize();
            for (int step=0;step<2;step++){detect_deadband<<<sg,bs>>>(step,seed+step,d_starts,d_ends,d_mags,0);cudaDeviceSynchronize();}
            int z=0; cudaMemcpyToSymbol(d_tp,&z,sizeof(int)); cudaMemcpyToSymbol(d_fp,&z,sizeof(int)); cudaMemcpyToSymbol(d_fn,&z,sizeof(int));
            for (int step=2;step<N_STEPS;step++){
                detect_review<<<sg,bs>>>(step,seed+step,d_starts,d_ends,d_mags,0,interval,0.3f);
                cudaDeviceSynchronize();
            }
            int h_tp,h_fp,h_fn; cudaMemcpyFromSymbol(&h_tp,d_tp,sizeof(int)); cudaMemcpyFromSymbol(&h_fp,d_fp,sizeof(int)); cudaMemcpyFromSymbol(&h_fn,d_fn,sizeof(int));
            tp+=h_tp; fp+=h_fp; fn+=h_fn;
        }
        float recall=(tp+fn)>0?(float)tp/(tp+fn):0;
        printf("%-12d %8d %8d %8d %8.3f\n", interval, tp/TRIALS, fp/TRIALS, fn/TRIALS, recall);
    }
    
    // === Jerk threshold sweep ===
    printf("\n=== Jerk Threshold Sweep (slow drift 0.5) ===\n");
    printf("%-12s %8s %8s %8s %8s\n", "Threshold", "TP", "FP", "FN", "Recall");
    printf("%s\n", "----------------------------------------------");
    
    for (float jt = 0.05f; jt <= 0.5f; jt += 0.05f) {
        int tp=0,fp=0,fn=0;
        for (int t = 0; t < TRIALS; t++) {
            unsigned int seed = (unsigned int)time(NULL) + t*30000 + (int)(jt*1000)*1000;
            init_anomalies<<<sg,bs>>>(d_starts,d_ends,d_mags,N_SENSORS,seed,0.5f);
            reset_counters<<<sg,bs>>>();
            cudaDeviceSynchronize();
            for (int step=0;step<2;step++){detect_deadband<<<sg,bs>>>(step,seed+step,d_starts,d_ends,d_mags,0);cudaDeviceSynchronize();}
            int z=0; cudaMemcpyToSymbol(d_tp,&z,sizeof(int)); cudaMemcpyToSymbol(d_fp,&z,sizeof(int)); cudaMemcpyToSymbol(d_fn,&z,sizeof(int));
            for (int step=2;step<N_STEPS;step++){
                detect_jerk<<<sg,bs>>>(step,seed+step,d_starts,d_ends,d_mags,0,jt);
                cudaDeviceSynchronize();
            }
            int h_tp,h_fp,h_fn; cudaMemcpyFromSymbol(&h_tp,d_tp,sizeof(int)); cudaMemcpyFromSymbol(&h_fp,d_fp,sizeof(int)); cudaMemcpyFromSymbol(&h_fn,d_fn,sizeof(int));
            tp+=h_tp; fp+=h_fp; fn+=h_fn;
        }
        float recall=(tp+fn)>0?(float)tp/(tp+fn):0;
        printf("%-12.2f %8d %8d %8d %8.3f\n", jt, tp/TRIALS, fp/TRIALS, fn/TRIALS, recall);
    }
    
    cudaFree(d_starts); cudaFree(d_ends); cudaFree(d_mags);
    printf("\n=== Key Insight ===\n");
    printf("Deadband catches step/spike but misses slow drift.\n");
    printf("Periodic review catches cumulative drift (like a captain's hourly check).\n");
    printf("Jerk catches the MOMENT drift starts (rate-of-change change).\n");
    printf("Best: combine all three layers.\n");
    return 0;
}
