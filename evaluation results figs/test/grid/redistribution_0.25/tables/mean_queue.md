# Grid | OD redistribution σ=0.25

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 55.94 | 56.88 | **54.66** | 56.56 | 61.43 |
| MA2C | 51.29 | 49.90 | **49.71** | 50.26 | 49.82 |
| IQLL | 63.02 | 69.87 | 46.83 | **37.12** | 94.16 |
| PPO | 41.51 | 37.71 | 20.71 | 24.08 | **18.68** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
