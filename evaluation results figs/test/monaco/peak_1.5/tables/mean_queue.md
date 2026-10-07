# Monaco | Peak demand ×1.5

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **205.10** | 252.73 | 209.67 | 222.84 | 213.41 |
| MA2C | 55.59 | **46.30** | 85.31 | 122.51 | 110.16 |
| IQLL | 341.16 | **242.89** | 323.10 | 323.16 | 318.05 |
| PPO | 214.78 | 261.13 | 63.13 | 96.03 | **45.25** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
