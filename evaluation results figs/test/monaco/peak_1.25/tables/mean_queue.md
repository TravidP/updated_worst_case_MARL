# Monaco | Peak demand ×1.25

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **173.29** | 237.67 | 174.18 | 200.33 | 179.65 |
| MA2C | 22.21 | **20.63** | 21.35 | 44.68 | 55.63 |
| IQLL | 338.83 | **224.88** | 317.90 | 318.17 | 315.03 |
| PPO | 184.66 | 246.69 | 12.84 | 21.72 | **9.55** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
