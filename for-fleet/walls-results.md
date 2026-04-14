# Barrier Topology × DCS Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps
5 topologies: open, 1-wall, cross, grid, concentric rings
Agents use wall-sliding (try x, then y if blocked)

## Results

| Topology | NoDCS | DCS | DCS Lift |
|----------|-------|-----|----------|
| Open | 1284 | 1418 | +10% |
| 1-Wall+gap | 1350 | 1416 | +5% |
| Cross+gaps | 1319 | 1418 | +7% |
| Grid+gaps | 1432 | 1429 | neutral |
| Concentric | 1510 | 1452 | **-4%** |

## Law 51: Barriers Systematically Reduce DCS Value
- Open → concentric: DCS lift drops from +10% to -4%
- Root cause: Euclidean distance ≠ path distance around walls
- DCS says "go there" but walls block direct path
- More complex topology = more misleading DCS directions

## Law 52: Spatial Barriers Improve Individual Foraging
- Grid NoDCS: +12% vs Open NoDCS
- Concentric NoDCS: +18% vs Open NoDCS
- Barriers create channeling effects, increase local density near food
- Wall-sliding creates systematic search vs random walk
- This is the OPPOSITE of what you'd expect — obstacles help!

## Design Implication
Environments that help individual foragers (barriers, corridors)
are the SAME environments that hurt shared information (DCS).
Physical structure helps autonomous agents but hinders coordination.
