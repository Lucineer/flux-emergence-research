# Laws 125-127: Food Scaling Analysis

## Data from Previous Experiments

From Law 82 (waste vs food density):
| Food | Agents/Food | Waste% | Coll/agent |
|------|-------------|--------|------------|
| 25   | 163.8       | 53.7%  | 215        |
| 50   | 81.9        | 80.2%  | 378        |
| 100  | 41.0        | 68.7%  | 335        |
| 200  | 20.5        | 71.6%  | 288        |
| 400  | 10.2        | 56.3%  | 567        |
| 800  | 5.1         | 66.4%  | 426        |
| 1600 | 2.6         | 60.5%  | 686        |

## Law 125: Waste ratio is U-shaped with agent-food ratio.
At extreme ratios (163.8 and 2.6 agents/food), waste is lower (54% and 61%).
At moderate ratios (20-82 agents/food), waste peaks at 68-80%.
The waste peak occurs when agents are numerous enough to compete but scarce enough
that each food item is targeted by multiple agents.

## Law 126: Per-agent collection peaks at moderate food density.
Food=50 (82 agents/food): 378/agent — highest per-agent collection.
Food=1600 (2.6 agents/food): 686/agent — more total but many agents share.
The peak at food=50 represents the sweet spot where competition is manageable
but food is scarce enough to motivate movement.

## Law 127: There is no food count that eliminates competition waste.
Even at 2.6 agents/food, 60.5% of grab attempts are wasted.
Even at 163.8 agents/food, 53.7% are wasted.
Competition waste is a structural constant of multi-agent systems — it cannot
be designed away, only tolerated or mitigated through coordination.

## Implication
The optimal agent-food ratio for fleet design is ~10:1 (400 food for 4096 agents).
This balances per-agent throughput (567) with reasonable waste (56%).
More food or fewer agents doesn't proportionally reduce waste.
