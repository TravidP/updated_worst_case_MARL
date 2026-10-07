# Monaco | Peak demand ×1.1

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **158.55** | 235.69 | 160.47 | 185.05 | 170.56 |
| MA2C | 18.22 | **17.55** | 18.26 | 36.11 | 21.93 |
| IQLL | 338.15 | **218.66** | 316.97 | 317.13 | 311.43 |
| PPO | 171.77 | 242.81 | 11.06 | 8.64 | **8.38** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
