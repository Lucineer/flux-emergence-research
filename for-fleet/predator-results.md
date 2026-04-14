# Predator-Prey × DCS Results

## Setup
2048 agents, 64 predators (faster, catch range=3), 400 food, grab=12, 3000 steps
Agents flee if predator within 20 cells; caught agents lose all food and respawn

## Results

| Mode | Agent Collection | Predator Kills |
|------|-----------------|----------------|
| NoDCS | 1110 | 0 |
| DCS | 2069 (+87%) | 0 |
| NoDCS+Pred | 22.5 | 7107 |
| DCS+Pred | 51.5 | 7461 |
| DCS+Pred-DCS | 40.2 | **9571** |

## Law 57: DCS is an Exploitable Information Channel

- Predators reduce agent collection by 98% regardless of DCS
- DCS barely helps under predation (22→51 vs 1110→2069 without predators)
- When predators FOLLOW DCS: 35% more kills (7461→9571)
- DCS becomes a targeting beacon for adversaries
- The communication channel that helps agents also helps predators
- Information sharing has an inherent security cost

## Design Implications
- Any shared communication channel is an attack surface
- Encrypted/steganographic DCS would be needed in adversarial environments
- In nature: ant pheromone trails are exploited by parasitic species
- The optimal DCS protocol is PRIVATE — only agents should read it
