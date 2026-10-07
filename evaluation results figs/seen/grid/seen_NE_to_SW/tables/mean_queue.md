# Grid | Northeast to southwest

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 114.45 | **97.88** | 154.68 | 155.86 | 149.26 |
| MA2C | **114.77** | 127.10 | 127.81 | 135.62 | 137.50 |
| IQLL | 263.32 | **119.13** | 191.49 | 186.32 | 194.53 |
| PPO | 101.06 | 91.53 | **59.66** | 97.36 | 65.25 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
