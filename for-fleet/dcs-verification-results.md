# DCS Verification Results

## Setup
4096 agents, 200 food, 1 guild, 3000 steps, migration speeds 0-4
Modes: NoDCS, DCS-Blind (follow stored point), DCS-Verify (check food near point first)

## Results

| Speed | NoDCS | DCS-Blind | DCS-Verify | Blind Lift | Verify Lift |
|-------|-------|-----------|------------|------------|-------------|
| 0 (static) | 1039 | 1164 | 1155 | **+12%** | **+11%** |
| 1 (slow) | 1220 | 1148 | 1171 | -6% | -4% |
| 2 (medium) | 1207 | 1057 | 1063 | -12% | -12% |
| 4 (fast) | 1171 | 995 | 1028 | -15% | -12% |

## Law 32: DCS with Moving Food is Fundamentally Broken

- Verification provides marginal improvement (+2-3%) over blind following
- Neither approach beats no-DCS when food migrates
- Root cause: sharing POSITION is wrong when targets move; need to share VELOCITY
- Verification costs extra scan without finding food (it already moved)
- The information format (x,y coordinates) assumes static targets

## Design Implication
For dynamic environments, DCS should share predicted future positions (x+vx*t, y+vy*t),
not past positions. This requires agents to estimate food movement velocity.
