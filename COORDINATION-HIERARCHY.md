# The Coordination Hierarchy: 73 Laws of Multi-Agent Information Sharing

## The Hierarchy (Robustness Order)

```
┌─────────────────────────────────────────────┐
│  1. INDIVIDUAL PERCEPTION                    │
│     Always works. No channel needed.         │
│     Limited by range, uncanny valley (L68-69)│
│     +0% baseline                              │
├─────────────────────────────────────────────┤
│  2. PERSONAL MEMORY                          │
│     Private, no communication. +16% (L73)    │
│     Must be conditional/fallback (L73)       │
│     Conflicts with DCS (L72)                 │
├─────────────────────────────────────────────┤
│  3. STIGMERGY (Environmental Traces)         │
│     Passive observation. +15-42% (L61)       │
│     Works with migration (L64: +27%)         │
│     Stacks with DCS (L62: +50%)              │
│     MORE noise-fragile than DCS (L63)        │
│     Minimal signal optimal (L67)             │
│     Optimal decay avoids stale trails (L66)  │
├─────────────────────────────────────────────┤
│  4. DCS (Shared Communication)               │
│     Active broadcast. +25-73% (L60)          │
│     15 strict requirements (DCS-DESIGN-THEORY)│
│     Zero noise tolerance (L42: 5%→-52%)      │
│     Exploitable by adversaries (L57)         │
│     Creates temporal bursts (L53: 16.9x)     │
│     Spatial concentration (L56: 18.4x)       │
│     Stacks with stigmergy (L62)             │
│     Conflicts with memory (L72)              │
│     THE DCS PARADOX (L60): most valuable     │
│     when least needed                         │
└─────────────────────────────────────────────┘
```

## The Design Principle

**Invest in layers bottom-up.** Each layer provides diminishing returns
but increasing fragility:

1. Better perception (sensors, range) — always worth it
2. Personal memory (4-8 slots) — cheap, private, reliable
3. Stigmergy (environmental traces) — works in dynamic environments
4. DCS (shared communication) — only when conditions are perfect

## Key Contradictions Resolved

| Question | Answer | Law |
|----------|--------|-----|
| Does more perception always help? | No — uncanny valley + collapse | 68-69 |
| Does shared info always help? | No — DCS paradox | 60 |
| Do mechanisms stack? | Stig+DCS yes, Mem+DCS no | 62, 72 |
| Is stigmergy better than DCS? | More robust, less absolute | 61-65 |
| Is memory better than DCS? | More private, but less lift | 71-73 |
| What about predators? | Both channels exploitable | 57, 65 |
| What about noise? | Stigmergy more fragile | 42, 63 |
| What about migration? | Stigmergy works, DCS fails | 29, 64 |
| What about barriers? | Help foragers, hurt DCS | 51-52 |

## 73 Laws Summary

### Foundation (1-26): Individual Agent Behavior
Grab range, accumulation, information scarcity, forced proximity,
cooperation protocols, specialist advantage, routing, DCS baseline,
population, world density, speed, TOP-K, energy, inter-guild sharing,
instinct, DCS inverted-U, migration, guild count, perception cost,
DCS invalidation, velocity prediction, individual perception dominance,
perception × DCS, PRNG irrelevance, MUD arena

### DCS Deep Dive (27-60): Shared Information Fragility
DCS with instincts, DCS density, DCS migration, guild count,
perception cost cliff, DCS broken with moving food, individual > DCS,
DCS × perception, noise (ZERO tolerance), broadcast (irrelevant),
heterogeneity (destroys), respawn (inverse), food value (independent),
owner death (location-based), multi-guild strategies, seed dependence,
DCS design theory synthesis, barriers (help foragers, hurt DCS),
temporal bursts, staggered activation, repulsion, spatial concentration,
predator exploitation, entropy (near-max), scaling (non-monotonic),
THE DCS PARADOX

### Stigmergy & Memory (61-73): Alternative Coordination
Stigmergy baseline, stigmergy+DCS stacking, stigmergy noise fragility,
stigmergy with migration, stigmergy under predation, stigmergy decay,
stigmergy deposit, perception uncanny valley, perception collapse,
personal memory, memory+DCS interference, memory conditional priority
