# Stigmergy Robustness Test Results

## Setup
4096 agents, 400 food, grab=12, 3000 steps

## Results

| Mode | Collection | vs None |
|------|-----------|---------|
| None | 1199 | baseline |
| DCS | 1495 | +25% |
| Stigmergy | 1377 | +15% |
| **DCS+Stigmergy** | **1804** | **+50%** |
| Stig+Noise5 | 263 | -78% |
| Stig+Noise20 | 44 | -96% |
| **Stig+Migrate** | **1528** | **+27%** |

## Law 62: DCS and Stigmergy Stack (+50% combined)
- DCS alone: +25%, Stig alone: +15%, Both: +50%
- Complementary information channels: locations + activity patterns
- 1 + 0.15 ≠ 1.50: synergy, not simple addition

## Law 63: Stigmergy More Fragile to Noise Than DCS
- 5% noise: Stig -78% vs DCS -52%
- Heat maps amplify noise through gradient calculations
- Even more sensitive to corrupted environmental data

## Law 64: Stigmergy Works With Migration Where DCS Fails
- Stigmergy +27% with migration vs +15% static
- DCS -6% to -22% with migration (Law 29)
- Moving food creates broader, more useful heat trails
- Stigmergy is the superior coordination mechanism for dynamic environments
