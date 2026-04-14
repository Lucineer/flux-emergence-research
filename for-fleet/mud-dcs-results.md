# MUD Arena with DCS Results

## Setup
1024 agents, 256 rooms, 32 scripts, 200 generations, 200 turns/gen
Agents share room gold info via ring-buffer DCS (1 point per guild)

## Results
- NoDCS: best=1743, avg=1679
- DCS: best=1714, avg=1691
- DCS lift: 0.98x (best), 1.01x (avg)

## Finding: DCS Neutral in MUD Arena

DCS provides no benefit because:
1. Room gold depletes (unlike respawning food) — by the time agents arrive, gold is gone
2. Agents naturally spread across rooms via genetic evolution
3. Room-based movement cost makes DCS following expensive
4. DCS room quality info is stale by the time other agents use it

This confirms Law 3 (information only matters under scarcity) in a new context:
MUD rooms have transient resources, so shared information has no persistent value.
