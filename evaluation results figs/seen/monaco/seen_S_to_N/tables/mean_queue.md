# Monaco | South to north

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **53.81** | 64.19 | 62.76 | 58.24 | 56.69 |
| MA2C | 37.03 | **36.06** | 41.36 | 39.71 | 38.77 |
| IQLL | 89.17 | 101.22 | 81.74 | 89.69 | **51.12** |
| PPO | 60.56 | 70.85 | **25.42** | 70.03 | 44.46 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
