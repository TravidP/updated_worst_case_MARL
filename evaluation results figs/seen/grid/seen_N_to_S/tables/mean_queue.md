# Grid | North to south

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **64.71** | 65.07 | 84.56 | 75.86 | 78.73 |
| MA2C | **63.85** | 65.86 | 84.02 | 80.38 | 83.35 |
| IQLL | 150.76 | **93.09** | 160.26 | 114.75 | 191.73 |
| PPO | 44.46 | 48.57 | **38.75** | 45.51 | 43.57 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
