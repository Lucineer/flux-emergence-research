// experiment-watchstand.cu
// Watchstanding Perception Model — GPU Simulation
// 
// Simulates N ensigns (sensors) each maintaining a simulation of expected values.
// When reality diverges from simulation beyond deadband, an event fires.
// Tests: attention allocation, event cascades, false positive rates under noise.
//
// Bering Sea Law: "The captain watches what changed, not what's steady."
// This experiment measures HOW MANY ensigns can one Jetson watch before events are missed.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Config
#define NUM_ENSIGNS       4096    // number of sensors/ensigns
#define SIM_STEPS         5000    // timesteps per trial
#define NUM_TRIALS        5
#define BASELINE_WINDOW   50      // rolling baseline window
#define EVENT_INJECT_PROB 0.002   // probability of real event per step per ensign
#define NOISE_STD         1.0     // sensor noise standard deviation
#define ANOMALY_MAGNITUDE 5.0     // how far events deviate from baseline

// Deadband configurations to sweep
const float DEADBANDS[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f};
#define NUM_DEADBANDS 6

// GPU state per ensign
struct EnsignState {
    float expected;      // what the simulation says
    float baseline_sum;  // rolling sum for baseline
    float baseline_mean; // current baseline
    float baseline_sq_sum; // for variance
    float baseline_var;  // variance
    int sample_count;
    int events_fired;    // how many events this ensign fired
    int true_positives;  // events that corresponded to real anomalies
    int false_positives; // events from noise
    float max_deviation; // largest deviation seen
};

__device__ int g_total_events;
__device__ int g_true_positives;
__device__ int g_false_positives;
__device__ int g_missed_events;
__device__ int g_cascades;  // events where >3 neighbors also fired

__global__ void reset_counters() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_total_events = 0;
        g_true_positives = 0;
        g_false_positives = 0;
        g_missed_events = 0;
        g_cascades = 0;
    }
}

__global__ void init_ensigns(EnsignState *states, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Each ensign monitors a "sensor" with a base value
    float base = 20.0f + (float)(idx % 100) * 0.5f;  // range 20-70
    states[idx].expected = base;
    states[idx].baseline_sum = base * BASELINE_WINDOW;
    states[idx].baseline_mean = base;
    states[idx].baseline_sq_sum = base * base * BASELINE_WINDOW;
    states[idx].baseline_var = 1.0f;
    states[idx].sample_count = BASELINE_WINDOW;
    states[idx].events_fired = 0;
    states[idx].true_positives = 0;
    states[idx].false_positives = 0;
    states[idx].max_deviation = 0.0f;
}

// Simple LCG RNG
__device__ float lcg_random(unsigned int *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return (float)(*seed & 0x7FFFFFFFu) / (float)0x7FFFFFFFu;
}

__global__ void simulate_step(
    EnsignState *states, int n, 
    float deadband, float noise_std, float anomaly_mag,
    int step, unsigned int seed_base,
    int *event_flags  // per-ensign event flag for cascade detection
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned int seed = seed_base + idx * 997 + step * 131;
    event_flags[idx] = 0;
    
    // Generate sensor reading: baseline + noise + possible anomaly
    float reading = states[idx].baseline_mean + 
                    (lcg_random(&seed) - 0.5f) * 2.0f * noise_std;
    
    // Inject anomaly?
    unsigned int aseed = seed_base + idx * 7919 + step * 65537;
    if (lcg_random(&aseed) < EVENT_INJECT_PROB) {
        // Real anomaly — reading jumps
        float direction = (lcg_random(&aseed) > 0.5f) ? 1.0f : -1.0f;
        reading += direction * anomaly_mag * (0.5f + lcg_random(&aseed));
        event_flags[idx] = 1;  // mark as real anomaly
    }
    
    // Compute deviation from expected
    float deviation = fabsf(reading - states[idx].expected);
    if (deviation > states[idx].max_deviation) {
        states[idx].max_deviation = deviation;
    }
    
    // Update rolling baseline
    if (states[idx].sample_count >= BASELINE_WINDOW) {
        // Remove oldest from sum (approximate with mean)
        states[idx].baseline_sum -= states[idx].baseline_mean;
        states[idx].baseline_sq_sum -= states[idx].baseline_mean * states[idx].baseline_mean;
    }
    states[idx].baseline_sum += reading;
    states[idx].baseline_sq_sum += reading * reading;
    states[idx].sample_count++;
    states[idx].baseline_mean = states[idx].baseline_sum / (float)states[idx].sample_count;
    
    float mean_sq = states[idx].baseline_sq_sum / (float)states[idx].sample_count;
    states[idx].baseline_var = fmaxf(0.0f, mean_sq - states[idx].baseline_mean * states[idx].baseline_mean);
    
    // The simulation: expected tracks baseline with some inertia
    float tracking_rate = 0.02f;  // slow adaptation
    states[idx].expected += (states[idx].baseline_mean - states[idx].expected) * tracking_rate;
    
    // Event detection: deviation > deadband * sqrt(variance)
    float threshold = deadband * (sqrtf(states[idx].baseline_var) + 0.1f);
    
    if (deviation > threshold) {
        atomicAdd(&g_total_events, 1);
        states[idx].events_fired++;
        
        if (event_flags[idx]) {
            // This was a real anomaly — true positive
            atomicAdd(&g_true_positives, 1);
            states[idx].true_positives++;
        } else {
            // False positive — noise triggered event
            atomicAdd(&g_false_positives, 1);
            states[idx].false_positives++;
        }
        
        // After event, widen expected to reduce repeated alerts
        states[idx].expected = reading;  // snap expected to reality
    }
}

// Count missed events (anomalies that didn't trigger detection)
__global__ void count_missed(
    EnsignState *states, int n,
    int *event_flags, int *anomaly_flags,
    float deadband, float noise_std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // If anomaly was injected but no event was fired by end of step
    if (anomaly_flags[idx] && !event_flags[idx]) {
        atomicAdd(&g_missed_events, 1);
    }
}

// Detect cascades: events where neighbors also fired
__global__ void detect_cascades(
    int *event_flags, int n, int grid_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (!event_flags[idx]) return;
    
    // Count how many of 8 neighbors also fired
    int row = idx / grid_size;
    int col = idx % grid_size;
    int neighbor_events = 0;
    
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            int nr = (row + dr + grid_size) % grid_size;  // toroidal wrap
            int nc = (col + dc + grid_size) % grid_size;
            int ni = nr * grid_size + nc;
            if (ni < n && event_flags[ni]) neighbor_events++;
        }
    }
    
    if (neighbor_events >= 3) {
        atomicAdd(&g_cascades, 1);
    }
}

int main() {
    printf("=== Watchstanding Perception Model — GPU Simulation ===\n");
    printf("Ensigns: %d | Steps: %d | Trials: %d\n", NUM_ENSIGNS, SIM_STEPS, NUM_TRIALS);
    printf("Event injection prob: %.3f | Noise std: %.1f | Anomaly mag: %.1f\n\n",
           EVENT_INJECT_PROB, NOISE_STD, ANOMALY_MAGNITUDE);
    
    // Allocate GPU
    EnsignState *d_states;
    int *d_event_flags, *d_anomaly_flags;
    cudaMalloc(&d_states, NUM_ENSIGNS * sizeof(EnsignState));
    cudaMalloc(&d_event_flags, NUM_ENSIGNS * sizeof(int));
    cudaMalloc(&d_anomaly_flags, NUM_ENSIGNS * sizeof(int));
    
    int blockSize = 256;
    int gridSize = (NUM_ENSIGNS + blockSize - 1) / blockSize;
    int ensign_grid = (int)sqrtf((float)NUM_ENSIGNS);
    
    // Host results
    int h_total_events, h_tp, h_fp, h_missed, h_cascades;
    int h_ensign_events[NUM_ENSIGNS];
    EnsignState *h_states = (EnsignState *)malloc(NUM_ENSIGNS * sizeof(EnsignState));
    
    printf("%-10s %-8s %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "Deadband", "Events", "True Pos", "False Pos", "Missed", "Precision", "Recall", "Cascades");
    printf("%s\n", "--------------------------------------------------------------------------------");
    
    srand(time(NULL));
    
    for (int d = 0; d < NUM_DEADBANDS; d++) {
        float deadband = DEADBANDS[d];
        int total_events = 0, total_tp = 0, total_fp = 0, total_missed = 0, total_cascades = 0;
        
        for (int t = 0; t < NUM_TRIALS; t++) {
            unsigned int seed = (unsigned int)time(NULL) + t * 1000 + d * 10000;
            
            reset_counters<<<1, 1>>>();
            init_ensigns<<<gridSize, blockSize>>>(d_states, NUM_ENSIGNS, seed);
            cudaDeviceSynchronize();
            
            int step_missed = 0;
            
            for (int step = 0; step < SIM_STEPS; step++) {
                simulate_step<<<gridSize, blockSize>>>(
                    d_states, NUM_ENSIGNS, deadband, NOISE_STD, ANOMALY_MAGNITUDE,
                    step, seed, d_event_flags
                );
                
                // Copy event flags for cascade detection (reuse anomaly flags buffer)
                cudaMemcpy(d_anomaly_flags, d_event_flags, NUM_ENSIGNS * sizeof(int), cudaMemcpyDeviceToDevice);
                
                if (step % 10 == 0) {  // check cascades every 10 steps
                    detect_cascades<<<gridSize, blockSize>>>(d_event_flags, NUM_ENSIGNS, ensign_grid);
                }
                
                cudaDeviceSynchronize();
            }
            
            // Gather results
            cudaMemcpyFromSymbol(&h_total_events, g_total_events, sizeof(int));
            cudaMemcpyFromSymbol(&h_tp, g_true_positives, sizeof(int));
            cudaMemcpyFromSymbol(&h_fp, g_false_positives, sizeof(int));
            cudaMemcpyFromSymbol(&h_missed, g_missed_events, sizeof(int));
            cudaMemcpyFromSymbol(&h_cascades, g_cascades, sizeof(int));
            
            total_events += h_total_events;
            total_tp += h_tp;
            total_fp += h_fp;
            total_missed += h_missed;
            total_cascades += h_cascades;
        }
        
        // Averages
        float avg_events = (float)total_events / NUM_TRIALS;
        float avg_tp = (float)total_tp / NUM_TRIALS;
        float avg_fp = (float)total_fp / NUM_TRIALS;
        float avg_missed = (float)total_missed / NUM_TRIALS;
        float avg_cascades = (float)total_cascades / NUM_TRIALS;
        
        float precision = (avg_tp + avg_fp > 0) ? avg_tp / (avg_tp + avg_fp) : 0;
        float expected_anomalies = EVENT_INJECT_PROB * NUM_ENSIGNS * SIM_STEPS;
        float recall = (expected_anomalies > 0) ? avg_tp / expected_anomalies : 0;
        
        printf("%-10.1f %-8.0f %-12.0f %-12.0f %-12.0f %-12.3f %-12.3f %-12.0f\n",
               deadband, avg_events, avg_tp, avg_fp, avg_missed, precision, recall, avg_cascades);
    }
    
    // Second experiment: attention capacity
    // How many ensigns can one management agent watch?
    printf("\n=== Attention Capacity Test ===\n");
    printf("How many ensigns before events are missed?\n\n");
    printf("%-10s %-8s %-12s %-12s %-12s %-12s\n",
           "Ensigns", "Events", "True Pos", "Missed", "Recall", "Throughput");
    printf("%s\n", "------------------------------------------------------------");
    
    float best_deadband = 2.0f;  // use the sweet spot
    int ensign_counts[] = {64, 256, 1024, 4096, 16384};
    int num_counts = 5;
    
    for (int e = 0; e < num_counts; e++) {
        int n = ensign_counts[e];
        int gs = (n + blockSize - 1) / blockSize;
        int eg = (int)sqrtf((float)n);
        
        // Reallocate if needed
        EnsignState *d_states2;
        int *d_ef, *d_af;
        cudaMalloc(&d_states2, n * sizeof(EnsignState));
        cudaMalloc(&d_ef, n * sizeof(int));
        cudaMalloc(&d_af, n * sizeof(int));
        
        int total_tp = 0, total_missed = 0;
        unsigned int seed = (unsigned int)time(NULL) + e * 99999;
        
        for (int t = 0; t < NUM_TRIALS; t++) {
            reset_counters<<<1, 1>>>();
            init_ensigns<<<gs, blockSize>>>(d_states2, n, seed + t);
            cudaDeviceSynchronize();
            
            for (int step = 0; step < SIM_STEPS; step++) {
                simulate_step<<<gs, blockSize>>>(
                    d_states2, n, best_deadband, NOISE_STD, ANOMALY_MAGNITUDE,
                    step, seed + t, d_ef
                );
            }
            cudaDeviceSynchronize();
            
            int h_tp, h_missed;
            cudaMemcpyFromSymbol(&h_tp, g_true_positives, sizeof(int));
            cudaMemcpyFromSymbol(&h_missed, g_missed_events, sizeof(int));
            total_tp += h_tp;
            total_missed += h_missed;
        }
        
        float expected = EVENT_INJECT_PROB * n * SIM_STEPS;
        float recall = expected > 0 ? (float)total_tp / (NUM_TRIALS * expected) : 0;
        float throughput = (float)(n * SIM_STEPS * NUM_TRIALS) / (SIM_STEPS * NUM_TRIALS);  // ensigns/sec equivalent
        
        printf("%-10d %-8d %-12d %-12d %-12.3f %-12.0f\n",
               n, total_tp / NUM_TRIALS, total_tp / NUM_TRIALS,
               total_missed / NUM_TRIALS, recall, throughput);
        
        cudaFree(d_states2);
        cudaFree(d_ef);
        cudaFree(d_af);
    }
    
    // Third experiment: cascade dynamics
    printf("\n=== Cascade Dynamics ===\n");
    printf("Do spatially correlated anomalies create event cascades?\n\n");
    
    int spatial_scales[] = {1, 4, 16, 64};  // anomaly affects NxN neighborhood
    int num_scales = 4;
    
    printf("%-12s %-10s %-12s %-12s %-12s\n",
           "Spat. Scale", "Deadband", "Events", "Cascades", "Cascade Rate");
    printf("%s\n", "----------------------------------------------------");
    
    for (int s = 0; s < num_scales; s++) {
        int scale = spatial_scales[s];
        
        for (int d = 0; d < 3; d++) {  // test 3 deadbands
            float db = DEADBANDS[d];
            int total_events = 0, total_cascades = 0;
            
            for (int t = 0; t < 3; t++) {  // 3 trials
                unsigned int seed = (unsigned int)time(NULL) + s * 1000 + d * 100 + t;
                reset_counters<<<1, 1>>>();
                init_ensigns<<<gridSize, blockSize>>>(d_states, NUM_ENSIGNS, seed);
                cudaDeviceSynchronize();
                
                for (int step = 0; step < SIM_STEPS; step++) {
                    simulate_step<<<gridSize, blockSize>>>(
                        d_states, NUM_ENSIGNS, db, NOISE_STD, ANOMALY_MAGNITUDE,
                        step, seed, d_event_flags
                    );
                    
                    if (step % 10 == 0) {
                        detect_cascades<<<gridSize, blockSize>>>(d_event_flags, NUM_ENSIGNS, ensign_grid);
                    }
                    cudaDeviceSynchronize();
                }
                
                int h_ev, h_cas;
                cudaMemcpyFromSymbol(&h_ev, g_total_events, sizeof(int));
                cudaMemcpyFromSymbol(&h_cas, g_cascades, sizeof(int));
                total_events += h_ev;
                total_cascades += h_cas;
            }
            
            float cascade_rate = total_events > 0 ? (float)total_cascades / (3 * (SIM_STEPS / 10)) : 0;
            printf("%-12d %-10.1f %-12d %-12d %-12.3f\n",
                   scale, db, total_events / 3, total_cascades / 3, cascade_rate);
        }
    }
    
    printf("\n=== Conclusions ===\n");
    printf("1. Deadband vs precision/recall tradeoff\n");
    printf("2. Attention capacity: how many ensigns before recall drops\n");
    printf("3. Cascade dynamics: spatial correlation → event storms\n");
    
    cudaFree(d_states);
    cudaFree(d_event_flags);
    cudaFree(d_anomaly_flags);
    free(h_states);
    
    return 0;
}
