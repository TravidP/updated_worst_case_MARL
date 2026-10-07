# Grid | Mixture scenario 1

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 85.96 | 85.52 | **85.36** | 94.33 | 134.43 |
| MA2C | 76.42 | 74.97 | **73.32** | 76.36 | 76.55 |
| IQLL | 132.62 | 110.83 | 130.90 | **100.67** | 136.88 |
| PPO | 66.68 | 58.33 | 27.11 | 33.79 | **24.18** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
