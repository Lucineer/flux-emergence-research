# Energy Budget × DCS Results

## Setup
4096 agents, 400 food, 256x256, grab=12, perc_cost=0.002, 3000 steps

## Results

| InitHP | NoDCS/agent | DCS/agent | Survival | DCS Lift |
|--------|-------------|-----------|----------|----------|
| 0.1 | 714 | 986 | 54% | **+38%** |
| 0.3 | 1079 | 1340 | 84% | +24% |
| 0.5 | 1224 | 1442 | 100% | +18% |
| 1.0 | 1120 | 1464 | 100% | **+31%** |
| 2.0 | 1040 | 1464 | 100% | **+41%** |
| 5.0 | 1116 | 1448 | 100% | +30% |

## Law 40: DCS Benefit is U-Shaped with Energy Budget

- Low energy (HP=0.1): DCS helps starving agents (+38%, 54% survive vs would-be worse)
- Medium energy (HP=0.5): DCS marginal (+18%, agents survive fine without it)
- High energy (HP=2.0): DCS most valuable (+41%, energy-rich agents exploit DCS fully)
- Pattern: DCS helps most when agents have energy to ACT on shared information
- Starving agents benefit from faster food-finding but die before reaching DCS targets
- Energy-rich agents follow DCS without survival pressure, maximizing its value
