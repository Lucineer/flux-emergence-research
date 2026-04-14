# Scarcity × Grab Range × DCS Three-Way Interaction

## Setup
4096 agents, 256x256 world, 3000 steps, 1 guild ring-buffer DCS
Sweep: 5 food levels × 3 grab ranges × 2 DCS conditions

## Results

| Food | Grab | NoDCS | DCS | DCS Lift |
|------|------|-------|-----|----------|
| 50 | 6 | 629 | 860 | 1.37x |
| 50 | 12 | 708 | 845 | 1.19x |
| **50** | **24** | **29** | **635** | **21.64x** |
| 100 | 6 | 605 | 1040 | 1.72x |
| 100 | 12 | 837 | 1027 | 1.23x |
| **100** | **24** | **31** | **988** | **32.24x** |
| 200 | 6 | 742 | 983 | 1.32x |
| 200 | 12 | 1035 | 1158 | 1.12x |
| 200 | 24 | 1243 | 1260 | 1.01x |
| 400 | 6 | 638 | 968 | 1.52x |
| 400 | 12 | 1309 | 1508 | 1.15x |
| 400 | 24 | 1489 | 1621 | 1.09x |
| 800 | 6 | 1055 | 1143 | 1.08x |
| 800 | 12 | 1070 | 1498 | 1.40x |
| 800 | 24 | 1517 | 1995 | 1.31x |

## Law 35: DCS Becomes Critical at Extreme Scarcity × Large Perception

- At food=50-100 with grab=24: DCS provides 21x-32x lift
- This is NOT useful DCS — it's a degenerate case where NoDCS agents starve
- Root cause: large grab range + scarce food + no DCS = agents wander randomly and starve
- DCS prevents starvation by providing ANY target to move toward
- Practical range: DCS lift of 1.1x-1.7x in non-degenerate conditions

## The DCS Sweet Spot
Maximum practical DCS benefit: food=100, grab=6 (+72%)
This combines scarcity (need to find food) with limited perception (can't find it alone)
