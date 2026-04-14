# CudaClaw Audit — JC1 Deep Mine

## What It Is
32K-line Rust+CUDA project for GPU-accelerated agent orchestration. Persistent CUDA kernels with warp-level parallelism, SmartCRDT state sync, lock-free CPU-GPU queues, DNA-based constraint system, NVRTC runtime kernel compilation.

## Key Innovations Worth Mining

### 1. Persistent Kernel Architecture (kernels/executor.cu)
- Warp 0 Lane 0 polls command queue, broadcasts via `__shfl_sync`
- Zero `cudaDeviceSynchronize()` in hot path
- `__threadfence_system()` for PCIe visibility
- Target: <5µs dispatch-to-execution
- **Relevance**: Our FLUX VM could use persistent kernel pattern instead of launch-per-instruction

### 2. DNA System (src/dna.rs)
- Hardware fingerprint + constraint theory + PTX muscle fibers + resource exhaustion metrics
- `.claw-dna` file = single source of truth for instance configuration
- **Relevance**: Our git-agent concept but at the GPU kernel level. DNA IS the config.

### 3. ML Feedback Loop (src/ml_feedback/)
- SuccessAnalyzer → MutationRecommendation → DNA Mutator
- Constraints can only relax 50%, can't tighten below observed minimums
- Version-tracked mutation history for rollback
- **Relevance**: Our emergence laws formalize what their feedback loop discovers empirically

### 4. SmartCRDT on GPU (kernels/executor.cu)
- LWW (Last-Writer-Wins) conflict resolution via atomicCAS
- Warp-parallel CRDT merge — 32 lanes each resolve conflicts on different slices
- **Relevance**: DCS stampede problem (Law 21) — LWW would give us automatic invalidation

### 5. NVRTC Runtime Compilation (src/ramify/nvrtc_compiler.rs)
- Compile CUDA kernels at runtime from PTX strings
- PTX branching for conditional kernel selection
- **Relevance**: "Bootstrap Bomb" — agents could compile their own optimized kernels

### 6. Constraint Theory (src/constraint_theory/)
- Geometric twin validation
- Safe bounds imported from constraint-theory project
- **Relevance**: Our energy cliff (Law 17) IS a constraint theory result

### 7. Muscle Fiber Concept (src/gpu_cell_agent/muscle_fiber.rs)
- Per-task specialized kernel configurations
- DNA stores PTX source + launch parameters for recompilation
- **Relevance**: Our DCS protocol could have muscle fibers — different kernels for different agent roles

## What's Missing (Our Laws Fill These Gaps)

1. **No emergence theory** — they build infrastructure but don't ask WHY certain patterns work
2. **No food/resource scarcity model** — all agents have equal access
3. **No spatial topology** — agents are in a flat queue, not rooms/grid
4. **No grab range / perception cost** — infinite perception by default
5. **No population scaling study** — claim 10K+ but no density-dependent effects
6. **CRDT has no invalidation** — stale data problem (our Law 21!)
7. **DNA mutation is random** — no directed evolution, no convergence-to-best (our Law 18)

## CUDA Experiments To Run

1. **Persistent kernel vs launch-per-step** — latency comparison on Jetson
2. **Warp-broadcast DCS** — use `__shfl_sync` instead of atomic global memory for guild info
3. **SmartCRDT food locations** — LWW on shared food map = automatic freshness
4. **NVRTC self-compilation** — agents write kernels that solve their own problem

## Actionable: Push to Our Stack

- Persistent kernel pattern → flux-vm next gen
- SmartCRDT → fix DCS stampede (Law 21) — LWW timestamp beats TTL
- DNA file format → git-agent `.vessel-dna` spec
- NVRTC → bootstrap bomb enabler

## Blockers
- Requires `cust` crate (Rust CUDA bindings) — not on Jetson
- Oracle1 can cross-compile Rust but can't test CUDA execution
- Need Oracle1 to build, JC1 to benchmark on real GPU
