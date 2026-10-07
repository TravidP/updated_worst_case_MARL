# Monaco | OD redistribution σ=0.75

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 128.20 | 224.97 | **106.04** | 132.48 | 131.64 |
| MA2C | 17.19 | **15.82** | 16.73 | 19.13 | 19.70 |
| IQLL | 339.02 | **227.38** | 309.51 | 321.79 | 310.07 |
| PPO | 147.41 | 238.96 | 10.26 | 7.96 | **7.57** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
