# Movement Speed × DCS Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps

## Results

| Speed | NoDCS/agent | DCS/agent | DCS Lift |
|-------|-------------|-----------|----------|
| 1 | 1193 | 1491 | **+25%** |
| 2 | 169 | 423 | **+151%** |
| 3 | 228 | 732 | **+221%** |
| 4 | 68 | 58 | -15% |
| 6 | 64 | 62 | -4% |
| 8 | 74 | 61 | -17% |

## Law 38: Movement Speed Has Inverted-U with DCS Benefit

- Speed 1 (slow): modest DCS benefit (+25%)
- Speed 2-3 (moderate): massive DCS benefit (+151% to +221%)
- Speed 4+ (fast): DCS hurts (-4% to -17%)
- Root cause: fast agents overshoot food targets, oscillating past them
- DCS compounds overshoot by directing agents to distant food they'll pass
- Optimal movement speed maximizes DCS value: fast enough to reach shared food, slow enough to grab it
