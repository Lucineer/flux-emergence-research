# MUD Team Competition Results

## Setup
2048 agents (1024/team), 256 rooms, 16 scripts/team, 200 generations, 200 turns/gen
Red vs Blue teams with per-team DCS ring buffer

## Results
- Red NoDCS: 3928, Red DCS: 3715 (-5%)
- Blue NoDCS: 3861, Blue DCS: 3811 (-1%)

## Finding: DCS Neutral-to-Negative in MUD Team Competition

Team competition doesn't make DCS valuable. The fundamental problem remains:
MUD room gold is transient (depletes on collection), so DCS shares stale info.
Neither cooperation nor competition changes the information staleness problem.
