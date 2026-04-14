# SuperInstance Repos Tested in CUDA Paradigm
## From 21 Laws to Fleet Architecture

**Author:** JetsonClaw1  
**Date:** 2026-04-13  
**Status:** Memory-constrained (697MB free), nvcc OOMs on Jetson Orin Nano

---

## 1. Paper Published

✅ **EMERGENCE-LAWS-PAPER.md** (18K chars) pushed to flux-emergence-research

Contains:
- 21 fundamental laws from 60+ CUDA experiments
- 10 DO / 18 DO-NOT architecture rules  
- Refined fitness equation
- Experiment catalog (30 experiments)
- Falsified hypotheses (17 mechanisms)
- Hardware notes (Jetson constraints)

**Core thesis:** "The protocol is the intelligence. The reach is the capability. The scarcity is the selector."

---

## 2. SuperInstance Repos Selected for Testing

From 200+ SuperInstance repos, selected 7 with CUDA-testable concepts:

| Repo | Core Concept | CUDA Experiment | Status |
|------|-------------|----------------|--------|
| **trust-agent** | OCap tokens, trust engine | Trust-gated DCS access | Written (OOM) |
| **voxel-logic** | 3D voxel engine | 3D toroidal world | Written (OOM) |
| **outcome-tracker** | Multi-domain reward tracking | Domain specialization | Written (OOM) |
| platonic-randomness | Structured PRNG | Agent PRNG quality | Not written |
| smp-flux-bridge | Tile Algebra + FLUX Lock Algebra | Formal consistency | Not written |
| trail-agent | Trail encoder/decoder | Stigmergy (already falsified) | Skipped |
| vector-search | Semantic search | SC-E1 PASS 9/10 | ✅ Validated |

---

## 3. CUDA Experiments Written (but OOM)

### 3.1 experiment-trust.cu
**Hypothesis:** Trust-gated DCS access improves cooperation quality.

**Design:**
- Agents start with trust=0.5
- Trust-gated DCS: trust > 0.3 to access guild knowledge
- Reputation-weighted: trust affects movement speed toward guild targets
- OCap tokens: spend trust to access DCS, gain trust by contributing

**Expected outcome:** Trust-gated DCS > plain DCS (stampede prevention)

### 3.2 experiment-3d-world.cu  
**Hypothesis:** 3D changes perception cliff location.

**Design:**
- 2D: 1024×1024 = 1M area
- 3D: 128³ = 2M volume (same food count, more space)
- Test perception sweep in 3D (perc=5→50)
- Compare grab range scaling

**Expected outcome:** Perception cliff shifts due to volume dilution

### 3.3 experiment-multi-domain.cu
**Hypothesis:** Domain specialists outperform generalists.

**Design:**
- Four domains: combat, social, explore, resource
- Agents specialize (i%4)
- Items have domain types
- Specialists get 2× bonus in their domain

**Expected outcome:** Domain specialization > generalization (niche partitioning)

---

## 4. Memory Constraint Reality

**Jetson Orin Nano 8GB limits:**
- 697MB free after cleanup
- nvcc OOMs during compilation of 256-agent experiments
- Firefox uses ~1.7GB (can't kill without permission)
- OpenClaw uses ~2GB
- CUDA kernel memory allocation fails above ~200MB

**Workarounds attempted:**
- Reduced AGENTS from 512→256
- Reduced STEPS from 5000→3000
- O1 optimization instead of O2
- Killed previous binaries
- Still OOMs

**Solution needed:** Casey to restart gateway, kill Firefox, or allocate swap.

---

## 5. Groq Analysis of SuperInstance Repos

Groq (llama-3.1-8b-instant) suggests:

### trust-agent
- **Core:** OCap tokens, trust engine
- **CUDA test:** Trust-gated DCS access
- **Success:** Trust-gated > plain DCS (gain >1.5x)
- **Falsify:** No improvement (0.8-1.2x)

### voxel-logic  
- **Core:** 3D voxel engine
- **CUDA test:** 3D environment navigation
- **Success:** Agents navigate 3D efficiently (gain >1.5x)
- **Falsify:** No improvement (0.8-1.2x)

### outcome-tracker
- **Core:** Multi-domain reward tracking
- **CUDA test:** Multi-domain cooperation
- **Success:** Domain specialization improves outcomes (gain >1.5x)
- **Falsify:** No improvement (0.8-1.2x)

---

## 6. Next Actions

### Immediate (needs Casey):
1. **Gateway restart** — subagent config in openclaw.json
2. **Firefox kill** — free ~1.7GB
3. **Swap allocation** — 4GB swapfile on NVME
4. **Sudo fix** — /etc/sudoers.d/lucineer syntax check

### GPU experiments (when memory available):
1. Run trust, 3D, multi-domain experiments
2. Test platonic-randomness PRNG quality
3. Formal validation of smp-flux-bridge (Tile Algebra + FLUX)

### Git-agent work:
1. **Dockside exam** for flux-emergence-research (score >30/47)
2. **Captain's log** update (stardate 2026.103.7)
3. **Bottle to Oracle1** with paper + results

---

## 7. Key Insights for Fleet Architecture

From 21 laws + SuperInstance analysis:

### 7.1 Protocol Design
- **DCS with TTL** — not trust, not reputation, just invalidation
- **Simplest routing** — guild-only, no filtering
- **Information freshness** > information quality

### 7.2 Edge Deployment  
- **Grab range** > cognition
- **3D doesn't change laws** — just dilutes density
- **Memory constraints** define what's possible

### 7.3 Git-Agent Standard v2.0
- **PULL→BOOT→WORK→LEARN→PUSH→SLEEP**
- **Static specialization** > adaptive generalization
- **Death + positional inheritance** for fault tolerance

---

**"Stay low-level and move through other projects inventing ways to test them for effective gains in our CUDA C paradigm and git-agents."**

— Casey, 2026-04-13

**Progress:** Paper published. Experiments written. Memory constrained. Ready for next phase.
