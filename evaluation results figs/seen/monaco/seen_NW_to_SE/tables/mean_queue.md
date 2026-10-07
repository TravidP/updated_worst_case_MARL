# Monaco | Northwest to southeast

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **31.49** | 37.68 | 32.98 | 35.01 | 34.93 |
| MA2C | 19.12 | **17.94** | 18.70 | 20.16 | 19.87 |
| IQLL | 109.17 | **57.01** | 113.89 | 128.36 | 108.36 |
| PPO | 25.13 | 29.19 | 17.11 | 16.87 | **14.83** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
