# Perception Cost Sweep Results

## Setup
4096 agents, 400 food, 3000 steps, no DCS, varying perception energy cost per step

## Results

| Cost | Collection/agent | Alive/4096 | Status |
|------|-----------------|------------|--------|
| 0.0000 | 942.4 | 4096 | All survive |
| 0.0010 | **1023.6** | 4096 | **Best** |
| 0.0020 | 945.2 | 4096 | All survive |
| 0.0040 | 915.9 | 4090 | Marginal |
| 0.0080 | 899.2 | 3282 | Dying |
| 0.0100 | 710.5 | 1934 | Dying |
| 0.0150 | 574.3 | 1187 | Dying |
| 0.0200 | 469.5 | 935 | Dying |
| 0.0300 | 1.2 | 0 | **Collapse** |
| 0.0500 | 1.1 | 0 | Collapse |
| 0.1000 | 1.1 | 0 | Collapse |

## Law 31: Perception Cost Has Gradual Cliff at ~0.03

- Cost 0-0.004: negligible impact, all survive
- Cost 0.001: **optimal** — +8.5% over free perception (cost filters bad scans)
- Cost 0.008-0.02: gradual die-off (20-80% survive)
- Cost ≥0.03: total population collapse
- Unlike energy cost (sharp cliff at 0.03, Law 17), perception has gradual decline
- Sweet spot: small nonzero cost improves efficiency by filtering unnecessary scans
