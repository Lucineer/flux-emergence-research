# Spawn Clustering × DCS Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps
Random spawn vs clustered spawn (16x16 jitter from center)

## Results

| Spawn | NoDCS | DCS | DCS Lift |
|-------|-------|-----|----------|
| Random | 1278 | 1453 | +14% |
| Cluster | 1291 | 1506 | +17% |
| Cluster advantage | +1% | +4% | — |

## Finding: Initial Spawn Position Doesn't Matter

- Clustering provides only +1% (NoDCS) to +4% (DCS) advantage
- Agents spread out within ~200 steps regardless of initial position
- Law 4 (forced proximity creates cooperation) requires PERSISTENT proximity
- Initial clustering is transient and provides no lasting benefit
