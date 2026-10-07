# Monaco | Northeast to southwest

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 16.19 | 17.03 | **12.42** | 13.65 | 13.34 |
| MA2C | 4.75 | 4.83 | **4.62** | 4.66 | 4.72 |
| IQLL | 100.38 | 79.28 | **73.74** | 84.53 | 96.97 |
| PPO | 9.00 | 13.75 | 2.97 | 2.22 | **1.81** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
