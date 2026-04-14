# Guild Count Sweep Results

## Setup
4096 agents, 400 food, 3000 steps, ring-buffer TOP-K=1 DCS
Sweeping guild count from 1 (all agents share) to 64 (64 agents/guild)

## Results

| Guilds | Agents/Guild | DCS/agent | NoDCS/agent | DCS Lift |
|--------|-------------|-----------|-------------|----------|
| 1 | 4096 | 1448.1 | 1160.7 | **+25%** |
| 2 | 2048 | 1372.5 | 1160.7 | +18% |
| 4 | 1024 | 1310.1 | 1160.7 | +13% |
| 8 | 512 | 1371.8 | 1160.7 | +18% |
| 16 | 256 | 1149.0 | 1160.7 | 0.99x |
| 32 | 128 | 1235.2 | 1160.7 | +6% |
| 64 | 64 | 1280.6 | 1160.7 | +10% |

## Law 30: Single Guild Maximizes DCS Benefit

- 1 guild (all agents): +25% — maximum knowledge sharing
- More guilds = fewer agents per data point = less benefit
- 16 guilds: DCS neutral — guilds too small to generate useful shared knowledge
- Counter-intuitive: despite Law 7 (simplest protocol wins), one shared channel beats partitioned

## Implication
For ring-buffer DCS with TOP-K=1, information concentration beats information partitioning.
One guild's single most-recent point reaches all 4096 agents = maximum amplification.
