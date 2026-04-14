# DCS Broadcast Frequency Results

## Setup
4096 agents, 400 food, 256x256, grab=12, 3000 steps
Intervals: 1 (every step) to 500 (every 500 steps)

## Results
All intervals: +6-9% DCS lift (statistically identical)

## Finding: Broadcast Frequency Irrelevant for Ring Buffer DCS

- DCS value doesn't depend on broadcast timing
- Ring buffer updates continuously via agent collection events
- In a large swarm (4096 agents), food is collected every step → always fresh data
- Broadcast frequency only matters in small populations or sparse environments
