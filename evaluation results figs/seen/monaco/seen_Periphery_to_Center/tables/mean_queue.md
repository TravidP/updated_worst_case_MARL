# Monaco | Periphery to center

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **66.29** | 115.27 | 93.09 | 72.22 | 72.53 |
| MA2C | 25.66 | 27.96 | 25.48 | **25.01** | 25.46 |
| IQLL | 235.28 | **55.94** | 234.89 | 229.67 | 228.94 |
| PPO | 46.10 | 132.96 | 13.82 | **9.06** | 10.09 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
