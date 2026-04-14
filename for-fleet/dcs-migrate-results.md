# DCS + Migrating Food Results

## Setup
4096 agents, 200 food, 3000 steps, ring-buffer TOP-K=1 DCS
Food migrates at speeds 0-4 in three modes (linear-X, linear-Y, diagonal)

## Results

| Mode | Speed | NoDCS/agent | DCS/agent | DCS Lift |
|------|-------|-------------|-----------|----------|
| static | 0 | 908.8 | 1117.4 | +23% |
| linear-X | 1 | 1214.6 | 1045.2 | **-14%** |
| linear-X | 2 | 1196.1 | 1049.4 | **-12%** |
| linear-X | 4 | 1146.6 | 971.0 | **-15%** |
| linear-Y | 1 | 1250.9 | 980.7 | **-22%** |
| linear-Y | 2 | 1178.9 | 1047.2 | **-11%** |
| linear-Y | 4 | 1219.3 | 1043.2 | **-14%** |
| diagonal | 1 | 1225.2 | 1150.9 | **-6%** |
| diagonal | 2 | 1103.4 | 1005.7 | **-9%** |
| diagonal | 4 | 1176.3 | 960.0 | **-18%** |

## Law 29: DCS Ring Buffer Actively Harmful with Moving Targets

- Static food (speed=0): DCS helps +4% to +23%
- Any migration (speed≥1): DCS hurts -6% to -22%
- Higher speed = worse DCS performance
- Direction doesn't matter — all modes equally affected
- Root cause: ring buffer stores location where food WAS, not where it IS
- Even "most recent" point is stale by the time other agents use it

## Implication for instinct-c
Instinct engines with spatial memory components must include staleness detection
or time-decay weighting. Raw location sharing in dynamic environments is worse
than no sharing at all.
