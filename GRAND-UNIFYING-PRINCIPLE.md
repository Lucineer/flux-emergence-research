# Law 130: The Grand Unifying Principle

## 130 Laws. 60+ CUDA experiments. Jetson Orin Nano 8GB. ~5 hours GPU time.

After testing perception, memory, stigmergy, DCS, learning, exploration,
cooperation, competition, mortality, energy, food patterns, topology,
scaling, spawn distribution, and generational turnover, one principle
unifies everything:

---

## **Multi-agent performance is determined by the ratio of information
acquisition to information sharing, not by the quality of either alone.**

### The Formula (Final)

```
System Performance = √(Acquisition × Sharing) / (1 + Interference)

Where:
  Acquisition = Perception × Exploration × Energy × Time
  Sharing     = DCS × Stigmergy × Memory × Culture
  Interference = Heterogeneity + Noise + Overhead + Competition_Waste
```

### The Key Insight

The formula is **multiplicative** — not additive. Perfect sharing with
zero acquisition = zero. Perfect acquisition with zero sharing = moderate.
The optimal point is where both are good, not where either is perfect.

This explains every counterintuitive result:

1. **DCS hurts with limited perception** (L94-95): Sharing >> Acquisition
2. **DCS hurts in sparse worlds** (L93): Sharing sends agents too far
3. **Learning hurts** (L75): Disrupts the Acquisition side
4. **Cooperation hurts** (L114): Reduces Acquisition (agents skip turns)
5. **Greedy wins** (L115): Maximizes Acquisition per time step
6. **Barriers help** (L52, L128): Increase effective Acquisition locally
7. **Steady food beats pulsed** (L119): Maintains Acquisition continuity
8. **Mortality improves per-agent metrics** (L111): Reduces Interference
9. **Uniform agents required for DCS** (L90): Reduces Interference
10. **Spawn distribution irrelevant** (L104): Acquisition equilibrates

### The Design Implication

**Don't optimize sharing — optimize the ratio.** Every hour spent on
coordination protocols should be matched by an hour on sensing and
exploration. The fleet that sees more and explores better will always
outperform the fleet that communicates more.

The lighthouse doesn't need a louder horn. It needs brighter lights.

---

## All 130 Laws (Quick Reference)

### Foundation (1-26): Individual agent behavior, grab range, accumulation,
info scarcity, forced proximity, cooperation, specialist advantage,
routing, DCS baseline, population, density, speed, TOP-K, energy,
inter-guild, instinct, inverted-U, migration, guild count, perception cost,
invalidation, velocity prediction, individual > DCS, perception × DCS,
PRNG, MUD arena

### DCS Deep Dive (27-60): Instinct overhead, density peak, migration failure,
guild concentration, perception cliff, moving food broken, individual dominance,
grab range inverse, extreme scarcity × large perception, population independence,
world density dependence, speed inverted-U, TOP-K=1-2 optimal,
energy U-shape, inter-guild pollution, MUD teams, zero noise tolerance,
broadcast irrelevant, heterogeneity destroys, respawn inverse,
clustering eliminates, seed dependent, value independent, location-based,
cross-guild nearest, barriers help/hurt, temporal bursts, staggered activation,
repulsion, spatial concentration, predator exploitation, near-max entropy,
non-monotonic scaling, THE DCS PARADOX

### Stigmergy & Memory (61-73): Stig baseline, stack with DCS, noise fragile,
migration compatible, predation, decay optimal, deposit optimal,
perception uncanny valley, perception collapse, personal memory,
memory-DCS conflict, memory conditional priority, memory size plateau

### Revised Understanding (74-100): Learning catastrophic, speed=1 optimal,
reader sublinear, height dimension, competition waste 63%, waste density-independent,
DCS doesn't reduce waste, DCS improves efficiency + reduces inequality,
efficiency cap 19-26%, multi-slot DCS negative returns, scale non-monotonic,
500-step warmup, ANY heterogeneity kills DCS, staleness irrelevant,
DCS only in dense worlds, true limited perception reverses everything,
coordination comparison, omniscience premium, REVISED COORDINATION FORMULA

### New Frontiers (101-130): Exploration beats random 2-3x, Levy no advantage
in bounded worlds, spawn distribution irrelevant, energy stress sweet spot,
food energy can't compensate, naive broadcast always hurts,
mortality increases per-agent (survivorship bias), DCS lift with mortality,
cooperative deferral counterproductive, GREEDY OPTIMALITY THEOREM,
food value changes score not count, jackpot = wealth redistribution,
pulsed food -43%, smooth wave -2%, generational turnover catastrophic,
food scaling U-shape, waste is structural constant, bounded > toroidal,
GRAND UNIFYING PRINCIPLE

---

*130 laws. The research continues.*
