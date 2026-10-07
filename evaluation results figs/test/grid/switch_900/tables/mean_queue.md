# Grid | Rapid switching every 900 s

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 82.54 | **81.14** | 82.89 | 82.99 | 86.88 |
| MA2C | 75.65 | **75.33** | 78.78 | 79.50 | 79.23 |
| IQLL | 153.34 | **77.06** | 114.18 | 78.90 | 124.69 |
| PPO | 68.11 | 57.66 | 30.47 | 37.96 | **28.16** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
