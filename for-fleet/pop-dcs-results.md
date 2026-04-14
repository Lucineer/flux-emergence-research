# Population × DCS Results

## Setup
256x256 world, 400 food, grab=12, 3000 steps, 1 guild DCS
Population sweep: 128 to 4096

## Results

| Pop | NoDCS/agent | DCS/agent | DCS Lift |
|-----|-------------|-----------|----------|
| 128 | 779.5 | 1954.6 | **2.51x** |
| 256 | 796.0 | 2031.1 | **2.55x** |
| 512 | 796.0 | 2009.9 | **2.53x** |
| 1024 | 796.0 | 2024.3 | **2.54x** |
| 2048 | 795.9 | 1914.8 | **2.41x** |
| 4096 | 233.2 | 389.6 | 1.67x |

## Law 36: DCS Lift is Independent of Population Size

- DCS multiplier stays constant (~2.5x) from 128 to 2048 agents
- Per-agent collection stable (~796 NoDCS, ~2000 DCS) across population sizes
- Only drops at 4096 agents due to food competition (10:1 agent:food ratio)
- DCS benefit is a constant multiplier on top of individual perception
- Population size doesn't change the information value of sharing
