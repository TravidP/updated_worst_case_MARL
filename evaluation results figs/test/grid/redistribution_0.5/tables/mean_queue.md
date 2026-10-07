# Grid | OD redistribution σ=0.5

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 57.22 | 57.36 | **55.02** | 57.68 | 64.35 |
| MA2C | 52.01 | 50.50 | **50.07** | 50.38 | 50.27 |
| IQLL | 65.69 | 93.68 | 39.81 | **34.46** | 95.79 |
| PPO | 42.36 | 38.19 | 20.66 | 24.20 | **18.38** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
