# Grid | Southeast to northwest

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **129.30** | 136.73 | 133.57 | 154.56 | 157.30 |
| MA2C | 111.37 | **102.50** | 130.00 | 142.25 | 141.43 |
| IQLL | 188.97 | 202.04 | **168.84** | 206.53 | 276.61 |
| PPO | 83.62 | 72.95 | **58.74** | 79.77 | 82.13 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
