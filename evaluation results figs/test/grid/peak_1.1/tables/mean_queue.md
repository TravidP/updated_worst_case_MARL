# Grid | Peak demand ×1.1

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 59.96 | 61.06 | **57.20** | 59.62 | 65.68 |
| MA2C | 54.60 | 52.68 | **52.19** | 53.00 | 52.44 |
| IQLL | 64.75 | 80.69 | 53.26 | **38.43** | 107.43 |
| PPO | 44.70 | 40.17 | 21.83 | 25.37 | **19.54** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
