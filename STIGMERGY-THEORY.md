# Stigmergy Theory: Environmental Coordination Without Communication

## Discovery (Laws 61-67)

Stigmergy — coordination through environmental traces — is a fundamentally different
coordination paradigm from DCS (shared communication).

### Key Properties

| Property | DCS | Stigmergy |
|----------|-----|-----------|
| Mechanism | Active communication | Passive observation |
| Information type | Specific location | Activity pattern |
| Channel security | Exploitable (Law 57) | Also exploitable (Law 65) |
| Migration tolerance | Fails (Law 29) | **Works** (Law 64: +27%) |
| Noise tolerance | 5% → -52% (Law 42) | 5% → -78% (Law 63) |
| Combination | N/A | **Stacks** with DCS (+50%, Law 62) |
| Optimal signal | N/A | Minimal (Law 67: deposit=1) |
| Decay rate | N/A | Non-trivial (Law 66: avoid stale trails) |
| Under predation | +16% (Law 65) | **+24%** (Law 65) |

### The Stigmergy Advantage

Stigmergy outperforms DCS in:
1. Dynamic environments (works with migration)
2. Adversarial environments (+24% vs +16% under predation)
3. Combined with DCS (complementary, not redundant)

Stigmergy fails worse than DCS in:
1. Noisy environments (-78% vs -52%)
2. Requires careful tuning (deposit amount, decay rate)

### The Perception Uncanny Valley (Laws 68-69)

More perception ≠ better performance:
- grab=4: 409, grab=6: 272 (more perception HURTS)
- grab=48: 58 (catastrophic collapse)
- Mechanism: perception overlap creates coordination overhead

### Design Principle

The optimal multi-agent system uses:
1. **Individual perception** (primary, works everywhere)
2. **Stigmergy** (secondary, for dynamic/adversarial environments)
3. **DCS** (tertiary, only when conditions are perfect)
4. **Minimal signals** (faint trails, low deposit, fast decay)

Perception is always > stigmergy > DCS in terms of robustness.
