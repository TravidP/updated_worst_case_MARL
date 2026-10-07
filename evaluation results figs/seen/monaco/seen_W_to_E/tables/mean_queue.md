# Monaco | West to east

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 44.02 | **36.16** | 45.01 | 44.06 | 44.28 |
| MA2C | **23.69** | 24.81 | 26.63 | 25.01 | 28.07 |
| IQLL | 104.44 | **66.39** | 112.33 | 126.85 | 97.19 |
| PPO | 40.51 | 40.46 | 15.97 | 37.56 | **11.70** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
