# Agent Heterogeneity × DCS Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps
Speed modes: uniform-slow(1), uniform-med(2), mixed(1-4), random(1-4)

## Results

| Mode | NoDCS | DCS | DCS Lift |
|------|-------|-----|----------|
| Uniform-slow | 1358 | 1483 | **+9%** |
| Uniform-med | 181 | 447 | **+147%** |
| Mixed-1-4 | 101 | 96 | **-5%** |
| Random-1-4 | 110 | 93 | **-16%** |

## Law 43: Agent Heterogeneity Destroys DCS Value

- Uniform populations: DCS helps (+9% to +147%)
- Heterogeneous populations: DCS hurts (-5% to -16%)
- Root cause: single DCS target can't serve mixed-speed agents
  - Fast agents overshoot, slow agents can't reach it in time
- Uniformity is a prerequisite for effective shared information
- This explains why biological systems with diverse individuals
  rely less on shared information and more on individual perception
