# DCS Noise Tolerance Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps
Noise: X% chance of adding ±X cell random offset to DCS point

## Results

| Noise% | Collection/agent | vs NoDCS |
|--------|-----------------|----------|
| 0 (no DCS) | 1267 | baseline |
| 1 (perfect) | 1487 | **+17%** |
| 5 | 607 | **-52%** |
| 10 | 613 | **-52%** |
| 20 | 567 | **-55%** |
| 30 | 540 | **-57%** |
| 50 | 492 | **-61%** |

## Law 42: DCS Has Zero Noise Tolerance

- Perfect DCS: +17%
- 5% noise: -52% (complete reversal!)
- 50% noise: -61%
- Even ±5 cells of positional error in a 256x256 world destroys DCS value
- DCS requires EXACT food location data — any corruption turns it into a liability
- Noisy shared information is WORSE than no shared information
- Design constraint: DCS only viable with perfect communication channels
