# Monaco | Uniform demand

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **125.18** | 222.08 | 133.82 | 177.03 | 131.54 |
| MA2C | 17.33 | **16.81** | 17.91 | 37.29 | 20.72 |
| IQLL | 334.16 | **229.22** | 312.14 | 319.61 | 307.41 |
| PPO | 160.40 | 228.68 | 10.32 | 7.86 | **7.59** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
