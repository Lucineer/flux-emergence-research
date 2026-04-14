# 80 Laws of Multi-Agent Emergent Coordination

## The Definitive Answer

After 80 fundamental laws and 50+ CUDA experiments on Jetson Orin GPU,
running 4096 agents across 256x256 toroidal worlds with 400 food items,
we can definitively answer the question: **how should multi-agent systems coordinate?**

### The Coordination Formula

```
Performance = Perception × (1 + Memory + Stigmergy + DCS) × (1 - Interference)
```

Where:
- **Perception** ( Laws 1, 34, 68-69): Individual sensing. Always first.
  - Limited by range, has uncanny valley, catastrophic collapse at extreme range
- **Memory** (Laws 71-74): Personal spatial memory. Conditional fallback only.
  - 2 slots optimal, +6%, conflicts with DCS, must be conditional
- **Stigmergy** (Laws 61-67): Environmental traces. Best for dynamic worlds.
  - +15-42%, works with migration, stacks with DCS, more noise-fragile
- **DCS** (Laws 2-60): Shared communication. Fragile, high-specificity.
  - +25-73% under ideal conditions, -6% to -96% under realistic conditions
  - 15 strict requirements, zero noise tolerance, exploitable, temporal bursts
- **Interference** (Laws 43, 53, 55, 72, 75): Negative interactions.
  - Heterogeneity, temporal bursts, repulsion, memory-DCS conflict, learning

### The Eight Principles

1. **Perception First** (Laws 1, 34, 68-69, 78-79)
   Individual sensing is always the highest-ROI investment.
   More perception helps until the uncanny valley (grab=6) and collapse (grab=48).

2. **Memory Second** (Laws 71-74)
   2-slot personal memory as conditional fallback. Never primary navigation.
   Conflicts with shared information channels.

3. **Stigmergy Third** (Laws 61-67)
   Environmental traces for dynamic/adversarial environments.
   Minimal signal (deposit=1), fast decay, works with migration.
   Stacks with DCS but more noise-fragile.

4. **DCS Last** (Laws 2-60)
   Only when ALL 15 conditions are met simultaneously:
   static food, dense world, uniform agents, perfect communication,
   fast respawn, scattered distribution, exact data, K=1-2, single guild,
   moderate speed, sufficient energy, no barriers, no noise,
   simultaneous access, no predators.

5. **The DCS Paradox** (Law 60)
   Information sharing is most valuable when least needed, and most needed
   when least valuable. The systems that NEED shared info can't USE it.

6. **Don't Learn, Optimize** (Laws 75-76)
   In multi-agent DCS environments, fixed strategies outperform adaptive ones.
   Learning creates oscillations; the optimal strategy is speed=1, always.

7. **Writers > Readers** (Law 77)
   DCS benefit scales sublinearly with reader fraction. 6% readers capture 83%
   of full benefit. One agent updating DCS helps all agents.

8. **Physical Structure Helps Autonomy, Hurts Coordination** (Laws 51-52)
   Barriers improve individual foraging (+18%) but reduce DCS (-4%).
   Obstacles create systematic search patterns but mislead shared directions.

### The Architecture

```
┌─────────────────────────────────────────────────────┐
│  AGENT                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │Perception │→ │  Memory  │→ │ Stigmergy Reader  │ │
│  │(sensors)  │  │(2 slots)│  │ (heat gradient)   │ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
│       ↓            ↓ (fallback)        ↓           │
│  ┌──────────────────────────────────────────────┐  │
│  │           Movement Controller                │  │
│  │  Speed=1 (fixed), wall-sliding, flee         │  │
│  └──────────────────────────────────────────────┘  │
│       ↓ (only if all above fail & conditions met)  │
│  ┌──────────────────────────────────────────────┐  │
│  │           DCS Reader (conditional)           │  │
│  │  Only when: static food, perfect comms, etc. │  │
│  └──────────────────────────────────────────────┘  │
│       ↓                                             │
│  ┌──────────────────────────────────────────────┐  │
│  │           Food Collection                     │  │
│  │  Grab range=12, update all coordination      │  │
│  │  channels on success                          │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### For Casey's Fleet

The coordination hierarchy applies directly to the Cocapn vessel fleet:
- Each vessel should have GOOD SENSORS (perception) first
- PERSONAL MEMORY (recent task history) as fallback
- STIGMERGY (environmental traces: file modifications, API calls) for coordination
- DCS (shared channels: Telegram, A2A) only for narrow, well-defined use cases
- NEVER use shared channels for dynamic/adversarial situations

The lighthouse doesn't need to shout — it needs to shine steadily.
