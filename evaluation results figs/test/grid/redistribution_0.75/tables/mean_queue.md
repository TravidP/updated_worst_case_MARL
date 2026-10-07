# Grid | OD redistribution σ=0.75

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 58.99 | 58.98 | **55.56** | 58.05 | 63.02 |
| MA2C | 52.99 | 51.33 | **50.84** | 51.33 | 51.04 |
| IQLL | 62.27 | 57.83 | 45.66 | **36.96** | 137.98 |
| PPO | 45.19 | 39.86 | 20.87 | 24.25 | **18.87** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
