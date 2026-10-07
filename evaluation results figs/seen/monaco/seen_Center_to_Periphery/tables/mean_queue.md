# Monaco | Center to periphery

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 76.14 | 104.34 | **74.89** | 88.83 | 93.94 |
| MA2C | **18.35** | 22.98 | 32.46 | 42.49 | 25.21 |
| IQLL | 160.99 | **68.04** | 125.14 | 155.09 | 150.35 |
| PPO | 102.32 | 108.85 | 91.92 | 106.17 | **17.13** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
