# Grid | Southwest to northeast

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 116.15 | **115.07** | 126.86 | 140.49 | 119.26 |
| MA2C | 126.13 | **120.70** | 131.98 | 128.56 | 129.46 |
| IQLL | 415.65 | **91.03** | 172.55 | 235.93 | 395.16 |
| PPO | 159.65 | 73.02 | 73.20 | 80.29 | **50.44** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
