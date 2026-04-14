# 100 Laws of Multi-Agent Emergent Coordination
## Century Milestone — April 14, 2026, Jetson Orin Nano 8GB

---

## The Revised Coordination Hierarchy (Post-Law 95)

The first 94 laws assumed **omniscient perception** (agents find nearest food globally).
Laws 94-98 introduced **true limited perception** and changed everything.

### Omniscient Perception (Laws 1-93)
```
DCS: +25-73% (under ideal conditions)
Stigmergy: +15-42%
Memory: +6-16%
Perception: baseline
```

### True Limited Perception (Laws 94-98)
```
Perc 12:  DCS +60% (blind → any info helps)
Perc 24:  No-coord wins (819 > 583 DCS > 502 stig)
Perc 48:  Stig+DCS +17% (only combo works)
Perc 96:  DCS -49% (shared info actively harmful)
```

## The Century Laws (99-100)

### Law 99: The Omniscience Premium
Global (omniscient) perception provides 10-14x more collection than true limited
perception at the same nominal range. The "perception" in Laws 1-93 was actually
omniscience — agents always found the nearest food globally. With true limited
perception at range 48: 890/agent (limited) vs ~5000/agent (omniscient).
The omniscience premium is the largest single factor in multi-agent performance.

### Law 100: The Revised Coordination Formula

```
Performance = Sensing × Exploration × (1 + Contextual_Sharing)
```

Where:
- **Sensing**: What agents can perceive RIGHT NOW (range × accuracy × freshness)
  - The omniscience premium (L99) dwarfs all coordination benefits
  - Investment in sensing always has highest ROI
- **Exploration**: How agents move when they can't perceive food
  - Random walk = absolute floor (L98: 60/agent, perception-independent)
  - Systematic exploration (spiral, grid) beats random by 5-10x
  - Individual exploration beats shared info at moderate perception (L97)
- **Contextual Sharing**: Information sharing that ACCOUNTS for perception limits
  - Only helps the nearly blind (L94: perc<12)
  - Must be spatially local, not global (L93: sparse worlds hurt)
  - Stigmergy+DCS synergy only at high perception (L96: perc 48+)
  - DCS creates travel waste in sparse worlds (L93: -57% at 512×512)

### The Eight Principles (Revised)

1. **Sensing First** — The omniscience premium (L99) is the biggest lever
2. **No Coordination > Bad Coordination** — At perc 24+, sharing hurts (L97)
3. **Coordination is Contextual** — Same mechanism helps at perc 12, hurts at perc 48
4. **Staleness Doesn't Matter** — DCS age 1 vs infinite: negligible (L91)
5. **Heterogeneity Kills** — Any capability difference destroys DCS (L90)
6. **Writers > Readers** — 6% readers capture 83% of benefit (L77)
7. **Learning Hurts** — Fixed > adaptive by 5.4x (L75)
8. **Efficiency Gap is Fundamental** — 74% of theoretical max is structurally lost (L85)

### The DCS Design Rules (Final)

DCS works ONLY when ALL of:
1. Dense world (density > 1.0) — L92
2. Omniscient or near-blind perception — L94-95
3. Uniform agent capabilities — L43, L90
4. Static or fast-respawning food — L44
5. Single shared point (not multi-slot) — L86
6. 500+ agents (critical mass) — L88
7. Perfect communication (zero noise) — L42

If ANY condition fails: DCS is neutral to severely harmful.

### Implications for the Cocapn Fleet

1. **Invest in sensors first** — every vessel needs maximum perception
2. **Default to no shared coordination** — individual exploration wins
3. **Use shared channels ONLY for well-bounded, well-understood use cases**
4. **Never use shared channels for dynamic/adversarial situations**
5. **Stigmergy (file changes, API logs) > DCS (messages) for fleet coordination**
6. **The lighthouse shines, it doesn't shout** — and vessels should look before they listen

---

*100 laws. 55+ CUDA experiments. 75+ commits. ~4 hours GPU time. Jetson Orin Nano 8GB.*
*All results in github.com/Lucineer/flux-emergence-research*
