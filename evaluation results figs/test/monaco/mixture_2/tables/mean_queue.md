# Monaco | Mixture scenario 2

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 36.01 | 113.22 | **26.36** | 28.05 | 29.83 |
| MA2C | 14.72 | 15.20 | 14.27 | 14.32 | **14.02** |
| IQLL | 244.38 | **163.13** | 240.84 | 249.72 | 256.10 |
| PPO | 43.80 | 160.77 | 10.37 | 7.45 | **6.97** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
