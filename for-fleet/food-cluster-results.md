# Food Clustering × DCS Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps

## Results

| Layout | NoDCS | DCS | DCS Lift |
|--------|-------|-----|----------|
| Scatter | 1237 | 1455 | **+18%** |
| 4-clust-r20 | 1504 | 1486 | neutral |
| 4-clust-r5 | 1504 | 1493 | neutral |
| 1-clust-r30 | 1498 | 1492 | neutral |
| 1-clust-r10 | 1497 | 1497 | neutral |

## Law 45: Food Clustering Eliminates DCS Value

- Scattered food: DCS +18% (normal benefit)
- Any clustering: DCS neutral (0.99-1.00x)
- Clustering raises NoDCS baseline (1237 → 1504) by making food easy to find
- DCS adds nothing because perception already finds clustered food
- DCS only valuable when resources are spatially dispersed
