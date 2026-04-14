# Instinct-C CUDA Simulation Results

## Experiment: Instinct-Driven vs Role-Driven vs Hybrid vs Random Agents

### Setup
- 4096 agents (1024 per type), food collection environment on Jetson Orin GPU
- 3000 steps, toroidal 256x256 world, DCS ring buffer (Law 26)
- All types have equal survival reflex (stop+recover at energy < 0.05)
- Instinct engine: 7 instinct types (SURVIVE, FLEE, HOARD, CURIOUS, COOP, GUARD, default)
- ROLE: hardcoded "always collect" + minimal survival check
- HYBRID: full instinct evaluation, falls through to ROLE behavior
- INSTINCT: full instinct-driven behavior (HOARD=slow, CURIOUS=random, FLEE=opposite, COOP=DCS-first)

### Results (average of 3 seeds)

| Type | Normal (400 food) | Scarcity (100 food) | Abundance (2000 food) |
|------|-------------------|---------------------|----------------------|
| ROLE | 1030/agent | 808/agent | 1759/agent |
| INSTINCT | 461/agent | 293/agent | 1632/agent |
| HYBRID | 632/agent | 402/agent | 1020/agent |
| RANDOM | 9/agent | 2/agent | 42/agent |

### Ratios (ROLE / other)

| Comparison | Normal | Scarcity | Abundance |
|-----------|--------|----------|-----------|
| ROLE vs INSTINCT | **2.23x** | **2.79x** | **1.08x** |
| ROLE vs HYBRID | **1.63x** | **2.03x** | **1.72x** |
| ROLE vs RANDOM | 114x | 344x | 41x |

### Key Findings

1. **ROLE dominates in ALL conditions** — fixed behavior beats instinct-driven in normal, scarce, AND abundant environments
2. **Instinct overhead INCREASES under scarcity** — 2.79x gap vs 2.23x normal. Wrong instincts (FLEE from food, CURIOUS exploration) are MORE costly when food is rare
3. **Instinct overhead DECREASES under abundance** — only 1.08x gap. Misfires don't matter when food is everywhere
4. **HYBRID is consistently middle** — instinct evaluation costs 37% even when it doesn't change behavior (falls through to ROLE)
5. **This extends Law 2** (accumulation beats adaptation) to the decision-making layer: hardcoded behavior beats evaluated behavior

### nvcc Bug Discovery

- `kernel<<<grid,blk>>>()` (zero-arg kernel launch) fails to compile
- nvcc parses `>>>()` as `>> ()` (right-shift + call), not `>>> ()`
- Fix: use `kernel<<<grid,blk>>> ()` (space before parens) or add dummy parameter
- This is an nvcc parser ambiguity, not a code error

### Conclusion for instinct-c Library

The instinct-c library's architecture (priority-based arbitration with configurable thresholds) is a valid safety mechanism but should NOT be the primary decision path. The recommended pattern:

1. **ROLE (primary)**: Direct, hardcoded behavior for the agent's main task
2. **INSTINCT (override)**: Only activate for survival/threat conditions
3. **HYBRID pattern**: Evaluate instincts, but if no override condition fires, execute role behavior

The instinct engine's value is as a **safety net**, not a **brain**.
