# Monaco | Mixture scenario 1

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 40.08 | 127.17 | **27.41** | 27.58 | 29.34 |
| MA2C | 15.49 | 15.79 | **14.80** | 15.26 | 14.95 |
| IQLL | 248.78 | **136.46** | 262.44 | 273.59 | 277.51 |
| PPO | 130.99 | 147.74 | 10.67 | 7.93 | **7.62** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
