# PRNG Quality Test Results

## Setup
4096 agents, 400 food, stochastic movement, 3000 steps
Three PRNGs tested: LCG (linear congruential), Xorshift32, MWC (multiply-with-carry)

## Results
- LCG: 55.3/agent
- Xorshift32: 55.4/agent
- MWC: 55.4/agent

## Conclusion: PRNG Choice Does Not Matter for These Simulations

All three generators produce statistically identical results. The stochastic jitter
component is small relative to deterministic movement, making PRNG quality irrelevant.
Our simple LCG is sufficient for all FLUX emergence experiments.
