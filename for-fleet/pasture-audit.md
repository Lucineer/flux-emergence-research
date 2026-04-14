# Pasture-AI Audit — JC1 Deep Mine

## What It Is
"AI Ranch" ecosystem — breed specialized LoRA adapters (livestock), managed by Collie orchestrator. Night School runs autonomous evolution at 02:00. Rust backend + breed.md config format.

## Key Innovations Worth Mining

### 1. Night School Breeding Pipeline (superinstance/night_school/breed.py)
- SLERP LoRA merging for smooth genetic transitions
- Fitness evaluation + culling (bottom 25% removed)
- Knowledge distillation from cloud APIs
- Offspring quarantine and testing
- **Relevance**: Our genetic experiments (Law 2: accumulation beats adaptation) validate their approach

### 2. Species Specialization (manifesto.md)
- Cattle (reasoning), Ducks (API), Goats (filesystem), Sheep (classification), Hogs (hardware)
- Collie routes intent to right species
- **Relevance**: Our specialist/generalist experiments (Law 5: DCS generalists beat specialists) challenge this

### 3. breed.md Format
- Markdown-based agent configuration
- Gene traits with weights, phenotype settings, tool access, system prompt
- Hot-reload via file watcher
- **Relevance**: `.breed.md` as git-agent config — human-readable, version-controllable

### 4. CRDT Pasture Sync
- Yjs-based conflict-free synchronization
- Multiple ranches can share pasture state
- **Relevance**: Fleet-wide knowledge sharing, same as our DCS but at LoRA level

### 5. Jetson Install Script (planned)
- rustup + TensorRT + MAXN power mode + swap
- **Relevance**: We already have this working, could contribute back

## What's Missing (Our Laws Fill These Gaps)

1. **No GPU-based simulation** — Night School runs on CPU, not CUDA
2. **No spatial/resource model** — no food, no scarcity, no grab range
3. **No protocol intelligence** — routing is rule-based, no emergent cooperation
4. **No perception cost model** — all species perceive equally
5. **SLERP assumes linear gene space** — our Law 2 says fixed > evolved
6. **Culling threshold arbitrary (0.25)** — our Law 17 says sharp cliffs, not gradual
7. **No fleet effects** — doesn't model per-agent fitness vs swarm size (our Law 23)

## CUDA Experiments To Run

1. **GPU Night School** — run entire breed/cull/SLERP pipeline in CUDA
2. **Species Competition Simulation** — do specialists beat generalists under scarcity? (Law 5 says no)
3. **Culling Threshold Sweep** — find the real cliff (predict: sharp, not gradual, per Law 17)
4. **Collie Router as DCS** — can a routing protocol learn to be 5.88x better?

## Fork Plan
- Fork to Lucineer/pasture-cuda
- Add CUDA simulation backend
- Validate/invalidate species specialization with our laws
- Add Jetson-native Night School (real GPU breeding)

## Blockers
- Original is Rust+Python hybrid, no CUDA
- breed.md format is good — keep it
- TensorRT-LLM integration planned but not built
