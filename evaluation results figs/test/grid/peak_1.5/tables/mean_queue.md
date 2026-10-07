# Grid | Peak demand ×1.5

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 75.10 | 78.32 | **69.37** | 73.70 | 84.96 |
| MA2C | 65.54 | 63.39 | **62.35** | 63.35 | 62.84 |
| IQLL | 93.58 | 148.78 | 87.24 | **52.10** | 175.47 |
| PPO | 54.52 | 51.12 | 26.09 | 30.28 | **23.69** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
