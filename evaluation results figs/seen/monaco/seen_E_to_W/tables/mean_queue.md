# Monaco | East to west

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 44.88 | 50.07 | **35.93** | 44.68 | 41.62 |
| MA2C | **20.74** | 21.22 | 20.97 | 21.50 | 21.31 |
| IQLL | 206.72 | 153.87 | **121.54** | 130.55 | 149.26 |
| PPO | 37.44 | 36.54 | 28.23 | **17.98** | 19.56 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
