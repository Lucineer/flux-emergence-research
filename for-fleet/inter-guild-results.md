# Inter-Guild DCS Sharing Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 8 guilds, 3000 steps

## Results

| Mode | Collection/agent | vs NoDCS |
|------|-----------------|----------|
| NoDCS | 839 | baseline |
| Own-guild | 1257 | **+50%** |
| All-guilds | 944 | +13% |
| All-guilds+nearest | 914 | +9% |

## Law 41: Inter-Guild DCS Sharing Creates Information Pollution

- Own-guild DCS: +50% (high relevance, spatially coherent)
- All-guilds DCS: +13% (diluted signal, distant guild data is noise)
- Checking more guild points DEGRADES performance by 37 percentage points
- Each guild's DCS point is spatially relevant only to that guild's agents
- Cross-guild sharing adds distant, irrelevant food locations
- More information ≠ better information; relevance > quantity

## Refinement of Law 30
Single guild is optimal not just because of concentration, but because
all stored points are spatially relevant to all agents using them.
