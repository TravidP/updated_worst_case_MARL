# Monaco | Rapid switching every 1200 s

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 40.69 | 87.62 | **38.90** | 59.23 | 41.15 |
| MA2C | 18.44 | 17.66 | **17.41** | 18.77 | 17.83 |
| IQLL | 290.12 | **132.97** | 279.57 | 270.41 | 265.21 |
| PPO | 41.21 | 134.94 | 11.92 | 11.04 | **8.88** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
