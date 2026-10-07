# Monaco | Southwest to northeast

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **44.64** | 48.53 | 47.14 | 47.18 | 45.43 |
| MA2C | 33.75 | **33.41** | 35.19 | 37.13 | 41.76 |
| IQLL | 96.23 | 56.24 | **54.51** | 107.65 | 67.72 |
| PPO | 36.09 | 41.03 | **30.20** | 31.47 | 63.61 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
