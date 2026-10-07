# Grid | Rapid switching every 300 s

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **113.01** | 122.54 | 119.57 | 128.33 | 151.01 |
| MA2C | 102.64 | **100.99** | 101.64 | 104.62 | 104.88 |
| IQLL | **118.07** | 138.40 | 185.32 | 202.13 | 160.20 |
| PPO | 82.00 | 84.54 | 36.24 | 55.47 | **32.99** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
