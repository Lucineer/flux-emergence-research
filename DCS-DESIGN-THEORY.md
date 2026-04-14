# DCS Design Theory: 50 Laws of Emergent Information Sharing

## The DCS Equation

After 50 fundamental laws and 100+ CUDA experiments on Jetson Orin GPU,
we can now state the conditions under which Distributed Cooperative Sharing (DCS)
creates value vs destroys it.

### DCS Value = f(environment) × g(agents) × h(protocol)

## Environment Requirements (MUST ALL be true)

| Requirement | Law | Condition |
|-------------|-----|-----------|
| Static resources | 29, 32, 33 | Food must not migrate (speed=0) |
| Fast respawn | 44 | Resources must replenish quickly (<50 steps) |
| Scattered distribution | 45 | Food must NOT be clustered |
| Dense world | 37 | Agents must be spatially close (density > 0.1) |
| Persistent resources | 45 | Room gold depletion kills DCS (MUD) |

## Agent Requirements (MUST ALL be true)

| Requirement | Law | Condition |
|-------------|-----|-----------|
| Uniform speed | 43 | All agents must move at same speed |
| Moderate speed | 38 | Speed 2-3 optimal; too fast = overshoot |
| Sufficient energy | 40 | U-shaped: very low or very high HP needed |
| Adequate perception | 34 | DCS inversely proportional to grab range |
| Small population per guild | 30 | 1 guild > 8 guilds > 16 guilds |

## Protocol Requirements (MUST ALL be true)

| Requirement | Law | Condition |
|-------------|-----|-----------|
| Exact data | 42 | ZERO noise tolerance (5% noise = -52%) |
| K=1-2 points | 39 | More stored points = worse |
| Single guild | 30, 41 | Cross-guild sharing = information pollution |
| No invalidation needed | 32 | TTL and verification don't help |
| Any broadcast frequency | freq | Frequency doesn't matter (continuously refreshed) |

## The DCS Fragility Map

DCS works ONLY when ALL 15 conditions above are met simultaneously.
Violating ANY single condition degrades or destroys DCS value.

### Conditions that DESTROY DCS (make it harmful):
1. Moving resources (Law 29: -6% to -22%)
2. Sparse world (Law 37: -42%)
3. Agent heterogeneity (Law 43: -5% to -16%)
4. Any noise (Law 42: -52%)
5. Cross-guild sharing (Law 41: -37pp vs own-guild)
6. Slow respawn (Law 44: -51%)
7. Fast movement (Law 38: -17%)

### Conditions that ELIMINATE DCS value (neutral):
1. Food clustering (Law 45: 0.99x)
2. Large grab range + abundance (Law 34: +10%)
3. Too many guilds (Law 30: neutral at 16)
4. TOP-K > 4 (Law 39: -15%)

## The Grand Insight

**DCS is a fragile, high-specificity protocol.** It provides +10-40% benefit
under ideal conditions but is harmful under most realistic conditions.

The reason: DCS shares LOCATION data about STATIC resources. Any deviation
from these assumptions (movement, noise, staleness, heterogeneity) turns
shared information from an asset into a liability.

## Design Implications for Real Systems

1. **Use DCS only when you can guarantee**: static resources, perfect communication,
   uniform agents, dense deployment
2. **Prefer individual perception** in most real-world conditions
3. **If using DCS**: single guild, K=1, no noise budget, verify resource stability
4. **For dynamic environments**: invest in individual perception, not shared information
5. **For heterogeneous teams**: each agent should use its own perception, not shared data

## The Paradox

The systems where DCS would be MOST useful (sparse, dynamic, heterogeneous environments)
are exactly the systems where DCS is MOST harmful. DCS only works in the "easy" cases
where individual perception already performs well.

This suggests that information sharing architectures should focus on
META-INFORMATION (strategies, models, policies) rather than raw DATA (locations, values).
