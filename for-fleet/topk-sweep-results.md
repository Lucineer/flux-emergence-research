# TOP-K DCS Sweep Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps, 1 guild ring buffer

## Results

| TOP-K | Collection/agent | vs K=0 |
|-------|-----------------|--------|
| 0 (none) | 1343 | baseline |
| 1 | 1512 | **+12.6%** |
| 2 | 1514 | **+12.7%** |
| 4 | 1140 | -15.1% |
| 8 | 945 | -29.7% |
| 16 | 915 | -31.9% |

## Law 39: TOP-K=1-2 is Optimal for DCS Ring Buffer

- K=1 and K=2 are statistically equivalent (+12.7%)
- K=4+ causes performance degradation (-15% to -32%)
- More stored points = more noise = worse routing decisions
- Agents checking 16 DCS points waste time on stale/distant locations
- Confirms and refines Law 26: the value is in recency, not quantity

## Practical Rule
Store 1-2 most recent food locations. Never store more than 4.
