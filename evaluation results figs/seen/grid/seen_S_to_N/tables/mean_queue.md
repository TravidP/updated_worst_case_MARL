# Grid | South to north

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **62.39** | 86.63 | 89.77 | 94.24 | 87.33 |
| MA2C | 64.04 | **63.20** | 88.17 | 82.45 | 87.22 |
| IQLL | 250.80 | 164.50 | 277.96 | 268.58 | **156.08** |
| PPO | 48.47 | **43.72** | 70.72 | 57.15 | 45.27 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
