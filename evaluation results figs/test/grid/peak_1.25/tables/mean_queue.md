# Grid | Peak demand ×1.25

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 65.46 | 66.42 | **62.13** | 64.87 | 71.30 |
| MA2C | 58.78 | 56.82 | **56.29** | 56.73 | 56.29 |
| IQLL | 68.22 | 105.72 | 65.64 | **45.32** | 127.90 |
| PPO | 48.40 | 43.86 | 23.22 | 27.21 | **21.25** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
