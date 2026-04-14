# DCS Velocity Prediction Results

## Setup
4096 agents, 200 food, diagonal migration, 1 guild, 3000 steps
Modes: NoDCS, Pos-DCS (share x,y), Vel-DCS (predict x+vx,y+vy), Vel+Verify (predict then check)

## Results

| Speed | NoDCS | Pos-DCS | Vel-DCS | Vel+Verify |
|-------|-------|---------|---------|-----------|
| 0 (static) | 1024 | 1082 | 1079 | **1109** |
| 1 (slow) | **1228** | 1089 | 1042 | 1067 |
| 2 (medium) | **1150** | 994 | 977 | 1020 |
| 4 (fast) | **1118** | 967 | 923 | 990 |

## Conclusion: No Information Sharing Strategy Beats Individual Perception for Moving Food

- Position sharing: -7% to -14% vs no-DCS
- Velocity prediction: WORSE than position (-13% to -17%)
- Verification: slight improvement over blind but still below no-DCS
- Individual perception is always best when targets move
- The information format doesn't matter — sharing ANY location data is harmful when targets are mobile

## Law 33: Individual Perception Dominates DCS for Mobile Targets

When targets move at speed≥1 per step, individual perception outperforms ALL shared
information strategies. DCS is only valuable for static or near-static resources.
