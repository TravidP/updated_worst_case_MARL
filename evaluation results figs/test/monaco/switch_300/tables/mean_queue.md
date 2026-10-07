# Monaco | Rapid switching every 300 s

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 55.08 | 89.97 | 52.45 | **47.37** | 55.05 |
| MA2C | 24.80 | 25.25 | **23.45** | 24.07 | 23.55 |
| IQLL | 187.70 | **145.89** | 167.84 | 210.17 | 215.04 |
| PPO | 72.94 | 97.11 | 30.20 | 26.49 | **17.26** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
