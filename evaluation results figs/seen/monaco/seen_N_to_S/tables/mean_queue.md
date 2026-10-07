# Monaco | North to south

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 12.17 | 15.32 | **11.57** | 12.06 | 11.85 |
| MA2C | 6.41 | 6.84 | 6.11 | 6.13 | **6.11** |
| IQLL | 84.98 | 74.22 | **64.24** | 78.69 | 95.07 |
| PPO | 10.09 | 15.13 | 4.96 | **3.03** | 3.35 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
