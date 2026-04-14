# DCS Invalidation Results

## Setup
4096 agents, 200 food, migration speeds 0-4, 1 guild, 3000 steps
Modes: NoDCS, DCS-no-invalidation, DCS-TTL20, DCS-TTL10, DCS-TTL5

## Results

| Migration | NoDCS | DCS-no-inval | DCS-TTL20 | DCS-TTL10 | DCS-TTL5 |
|-----------|-------|-------------|-----------|-----------|----------|
| Speed 0 (static) | 917 | **1140** (+24%) | 1136 (+24%) | 1136 (+24%) | 1136 (+24%) |
| Speed 1 (slow) | 1222 | 1126 (-8%) | 1163 (-5%) | 1160 (-5%) | — |
| Speed 2 (medium) | 1158 | 1079 (-7%) | 1047 (-10%) | 1061 (-8%) | — |
| Speed 4 (fast) | 1153 | 995 (-14%) | 962 (-17%) | 980 (-15%) | — |

## Key Finding: TTL Invalidation Does NOT Fix Moving-Food DCS Problem

- Static food: DCS +24%, TTL irrelevant (no staleness)
- Moving food: DCS hurts -7% to -17%, TTL invalidation makes it WORSE or neutral
- Root cause: TTL discards information that IS still somewhat useful
- The problem is LOCATION staleness (food moved), not TIME staleness (info expired)
- TTL can't distinguish "food moved" from "food still there but info is old"

## Design Implication
DCS needs location VERIFICATION, not time-based invalidation.
Before following DCS point, agent should check if food actually exists there.
