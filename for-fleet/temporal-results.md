# Temporal Collection Patterns

## Setup
4096 agents, 400 food, grab=12, 3000 steps
Tracked per-step collection count, autocorrelation at lag-50

## Results

| Metric | NoDCS | DCS |
|--------|-------|-----|
| Total collections | 5.03M | 6.03M |
| Avg per step | 1678 | 2009 |
| Variance | 888K | 859K |
| Autocorrelation (lag-50) | 0.050 | **0.103** |
| Burst ratio (max/min window) | 3.61x | **16.86x** |
| Min window rate | 500/step | 132/step |
| Max window rate | 1804/step | 2230/step |

## Law 53: DCS Creates Temporal Burst Patterns

- DCS doubles autocorrelation (0.050 → 0.103): rhythmic pulse emerges
- DCS quintuples burst ratio (3.6x → 16.9x): extreme feast-famine cycles
- Root cause: shared info synchronizes agent movement → mass convergence
- All agents rush to same DCS point → massive collection burst
- After burst: no food nearby → quiet period until new food respawns/DCS updates
- NoDCS agents spread out → steady collection, no synchronization

## Design Implications
- DCS causes resource oscillations that could destabilize ecosystems
- Temporal burstiness is a HIDDEN COST of shared information
- Systems with DCS need burst-dampening mechanisms (rate limits, staggered activation)
- This explains why natural swarms use pheromone DECAY — it prevents synchronization
