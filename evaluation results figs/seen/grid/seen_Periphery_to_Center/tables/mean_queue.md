# Grid | Periphery to center

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 59.55 | 61.99 | **57.30** | 58.99 | 63.26 |
| MA2C | 52.53 | 51.43 | **50.73** | 51.51 | 51.44 |
| IQLL | 70.47 | 63.39 | 87.67 | **28.38** | 33.26 |
| PPO | 47.37 | 43.38 | 20.72 | 24.39 | **18.94** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
