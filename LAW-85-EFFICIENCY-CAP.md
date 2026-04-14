# Law 85: The Multi-Agent Efficiency Cap

## Theoretical Analysis

With 400 food, 50-step respawn, 4096 agents, 3000 steps:
- Maximum food events: 400 × (3000/50) = 24,000
- Maximum per agent: 24,000/4096 = 5.86
- NoDCS actual: ~1.12 = **19% of theoretical maximum**
- DCS actual: ~1.50 = **26% of theoretical maximum**
- The gap: **74% of potential is lost**

## The Gap Decomposition

1. **Travel time** (~30%): Agents can't teleport to food
2. **Competition waste** (~60%): Multiple agents target same food (Law 81)
3. **Perception limit** (~5%): Agents can't see all food simultaneously
4. **Respawn timing** (~5%): Food not available between collection and respawn

## What Closes The Gap?

| Intervention | Gap Closed | Mechanism |
|-------------|-----------|-----------|
| DCS | +7pp | Faster convergence, not better targeting |
| Stigmergy | +6pp | Activity-based routing |
| More food | +20-40pp | Less competition |
| Fewer agents | +10-30pp | Less competition |
| Larger grab | variable | Perception uncanny valley (L68-69) |
| Speed increase | negative | Overshoot, temporal bursts |

## The Hard Truth

**No coordination mechanism closes more than ~7pp of the 81pp gap.** The remaining gap is fundamental to parallel agents competing for finite resources. Information sharing makes agents faster but not smarter about avoiding each other.

The only way to significantly close the gap: reduce agent count or increase resource density. This is the **agent-food ratio theorem** — performance scales with food-per-agent, and coordination mechanisms are second-order effects.

## Implication for Fleet Design

Don't over-invest in coordination protocols. The ROI is capped at ~7pp improvement.
Instead, ensure adequate resources (tasks, compute) per agent. An overstaffed system
with perfect coordination will always underperform an adequately-staffed system with no coordination.
