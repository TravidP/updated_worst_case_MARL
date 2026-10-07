# Monaco | Southeast to northwest

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 65.39 | **58.74** | 77.57 | 89.19 | 71.60 |
| MA2C | 36.76 | **32.18** | 33.59 | 57.00 | 41.44 |
| IQLL | 184.29 | **102.34** | 117.65 | 124.65 | 120.06 |
| PPO | 51.77 | 40.97 | **29.76** | 46.23 | 33.75 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
