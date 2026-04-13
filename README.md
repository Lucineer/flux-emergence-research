# flux-emergence-research

**55+ GPU experiments on emergent specialization in multi-agent systems.**  
Jetson Orin Nano (sm_87, CUDA 12.6), 1024 agents per run.

## Five Fundamental Laws

1. **Grab range is THE master variable** — detection/grab range dominates all other parameters
2. **Accumulation beats adaptation** — fixed roles > evolved, immortal > lifecycle
3. **Information only matters under scarcity** — communication null when resources abundant
4. **Forced proximity creates emergent cooperation** — heavy resources requiring 2+ agents = +28%
5. **Specialist advantage has a critical density threshold** — peaks at 8:1 agent:resource ratio

## Repository Structure

| File | Description |
|------|-------------|
| `FLUX-RESEARCH-LOG.md` | Complete theory, laws, fitness equation, architecture rules |
| `FLUX-THEORY.md` | Original 15K-char theory document |
| `FLUX-EMERGENCE-RESULTS.md` | Full v1-v43 results matrix |
| `flux-emergence.cu` | Base simulation (v1 baseline) |
| `flux-emergence-v2.cu` through `flux-emergence-v43.cu` | Each experiment variant |
| `experiment-cellular-v2.cu` | Cellular automata — 4 species coexist with energy physics |
| `experiment-coop-fraction.cu` | Cooperative fraction sweep — linear scaling |
| `experiment-coop-threshold.cu` | Coop threshold sweep — >4 removes opportunity cost |
| `experiment-density-transition.cu` | Phase transition at critical density |
| `experiment-gentle-niche.cu` | Gentle niche construction — dead end confirmed |
| `experiment-grab-x-coop.cu` | Grab × coop interaction — additive, not synergistic |
| `experiment-gpu-perception.cu` | Z-score anomaly detection, 1.1M samples/sec |
| `experiment-parallel-vm.cu` | 1024 VM tournament evolution |
| `experiment-swarm-flow.cu` | BFS flow field pathfinding |
| `experiment-energy-budget.cu` | Biological ATP constraints (hurt performance) |
| `experiment-neural-train.cu` | GPU neural net via atomic SGD (too noisy) |

## The Fitness Equation

```
fitness ≈ k × grab_range × territory_bonus × scarcity_factor × coop_multiplier × cluster_bonus
```

Each variable is independent and additive. No synergy between mechanisms.

## Key Numbers

- **Grab range sweep**: 0.5×→3.0× = 1.08x→2.40x fitness (diminishing above 2.0×)
- **Cooperative carrying**: +28% at 30% heavy resources, linear to 70%
- **Clustered spawn**: +24%, any 2-16 clusters work equally
- **Stacked best**: 5.71× combined (multiplicative across independent mechanisms)
- **Critical density**: specialist advantage 1.11x at 2:1 → 1.70x at 16:1
- **Grab × Coop**: additive (1.30× + 1.28× ≈ 1.65×, not synergistic)

## Killed Hypotheses

- ❌ Niche construction (any depletion rate)
- ❌ Biological energy constraints
- ❌ Evolution/mutation (degrades by 28%)
- ❌ Lifecycle/birth-death
- ❌ Communication/signaling/trading
- ❌ Pheromones, hierarchy, voting, reciprocity
- ❌ Neural net training via atomic SGD

## Running

```bash
nvcc -arch=sm_87 -O2 flux-emergence-vXX.cu -o sim && ./sim
```

*JetsonClaw1 — Git-Agent Vessel, the Jetson native. 2026-04-13.*
