# Grid | Mixture scenario 3

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **96.89** | 110.69 | 98.08 | 117.56 | 116.05 |
| MA2C | 92.42 | **87.68** | 88.32 | 97.20 | 98.38 |
| IQLL | **54.77** | 79.30 | 65.57 | 266.47 | 138.88 |
| PPO | 71.04 | 57.67 | 26.72 | 35.01 | **25.54** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
