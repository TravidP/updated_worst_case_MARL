# Monaco | OD redistribution σ=0.25

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 148.21 | 222.86 | **129.32** | 175.09 | 150.95 |
| MA2C | 17.22 | **16.08** | 17.32 | 19.79 | 18.21 |
| IQLL | 331.96 | **238.20** | 311.27 | 313.98 | 309.30 |
| PPO | 127.43 | 221.50 | 10.27 | 8.01 | **7.64** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
