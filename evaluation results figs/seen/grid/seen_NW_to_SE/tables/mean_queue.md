# Grid | Northwest to southeast

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 110.06 | **99.66** | 129.46 | 147.78 | 188.00 |
| MA2C | **97.37** | 106.40 | 118.83 | 123.35 | 123.46 |
| IQLL | 355.48 | 224.10 | 142.62 | **131.32** | 219.84 |
| PPO | 123.37 | 75.65 | **62.63** | 111.90 | 65.83 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
