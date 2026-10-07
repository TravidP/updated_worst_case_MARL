# Grid | Center to periphery

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 60.28 | 59.45 | **58.66** | 62.13 | 69.14 |
| MA2C | 52.21 | **50.88** | 53.11 | 53.68 | 53.56 |
| IQLL | 433.80 | 86.10 | 108.23 | **72.31** | 303.35 |
| PPO | 34.57 | 33.20 | 22.08 | 26.41 | **19.28** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
