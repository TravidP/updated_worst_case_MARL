# Grid | Rapid switching every 1200 s

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 134.49 | **100.36** | 100.60 | 111.77 | 110.37 |
| MA2C | 82.85 | **74.60** | 85.46 | 87.73 | 88.36 |
| IQLL | 471.41 | **234.35** | 446.85 | 289.13 | 420.17 |
| PPO | 138.29 | 49.91 | 36.26 | 39.91 | **27.80** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
