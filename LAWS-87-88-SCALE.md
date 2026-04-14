# Laws 87-88: DCS Scaling Behavior

## Experiment: 5 scales from 128 to 8192 agents

| Agents | Food | NoDCS | DCS | Lift |
|--------|------|-------|-----|------|
| 128 | 12 | 444 | 434 | 0.98x |
| 512 | 50 | 947 | 1256 | 1.33x |
| 2048 | 200 | 1233 | 1254 | 1.02x |
| 4096 | 400 | 708 | 977 | 1.38x |
| 8192 | 800 | 611 | 679 | 1.11x |

## Law 87: DCS lift is non-monotonic with system scale.
DCS doesn't smoothly scale — it has peaks at certain agent/food ratios.
Peak at 4096 agents (+38%), secondary peak at 512 (+33%).
Fails at 128 (too few agents to benefit from sharing) and weakens at 8192 (too much competition).

## Law 88: DCS requires critical mass (500+ agents) to provide benefit.
Below 128 agents, DCS is slightly negative (0.98x).
The overhead of shared communication exceeds the benefit when agent count is too low.
DCS is a large-swarm optimization, not a small-team coordination tool.
