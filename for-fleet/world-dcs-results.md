# World Size × DCS Results

## Setup
4096 agents, 400 food, grab=12, 3000 steps, 1 guild DCS
World sizes: 64 to 512 (density 1.1 to 0.017)

## Results

| World | Density | NoDCS/agent | DCS/agent | DCS Lift |
|-------|---------|-------------|-----------|----------|
| 64 | 1.098 | 1231 | 2337 | **+90%** |
| 128 | 0.274 | 2516 | 1946 | **-23%** |
| 256 | 0.069 | 2023 | 1328 | **-34%** |
| 512 | 0.017 | 1315 | 759 | **-42%** |

## Law 37: DCS Benefit Depends on World Density

- Dense world (64x64): DCS +90% — agents cluster, shared info is nearby
- Sparse worlds (128-512): DCS hurts -23% to -42%
- Root cause: single DCS point is far from most agents in sparse worlds
- Following distant DCS point wastes more movement than it saves
- DCS is a LOCAL information protocol — it works when agents are close together
- The agent:food ratio matters, but world density matters MORE

## Design Implication
DCS should be used when agents are spatially clustered (dense environments).
In sparse/distributed environments, individual perception dominates.
