# Monaco | Mixture scenario 3

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 37.84 | 73.89 | **32.79** | 36.54 | 38.21 |
| MA2C | 14.75 | 16.82 | 13.79 | 14.10 | **13.41** |
| IQLL | 180.83 | **126.09** | 161.47 | 190.96 | 196.50 |
| PPO | 45.58 | 47.57 | 10.66 | 9.06 | **8.35** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
