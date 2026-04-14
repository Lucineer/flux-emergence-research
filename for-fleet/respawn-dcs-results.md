# Food Respawn Rate × DCS Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps
Respawn times: 10 to 500 steps

## Results

| Respawn | NoDCS | DCS | DCS Lift |
|---------|-------|-----|----------|
| 10 (fast) | 1035 | 1722 | **+66%** |
| 25 | 1144 | 1643 | +44% |
| 50 (default) | 1252 | 1485 | +19% |
| 100 | 1031 | 1242 | +20% |
| 200 (slow) | 935 | 796 | -15% |
| 500 (very slow) | 352 | 173 | **-51%** |

## Law 44: DCS Benefit Inversely Proportional to Resource Turnover Time

- Fast respawn: DCS highly valuable (+66%) — shared points remain valid
- Slow respawn: DCS harmful (-51%) — shared points are stale (food collected, hasn't returned)
- Crossover at ~150 steps: DCS flips from helpful to harmful
- Resource persistence is a prerequisite for DCS value
- Combined with Law 42 (noise), Law 37 (density), Law 29 (migration):
  DCS requires PERSISTENT, STATIC, LOCAL, EXACT resources
