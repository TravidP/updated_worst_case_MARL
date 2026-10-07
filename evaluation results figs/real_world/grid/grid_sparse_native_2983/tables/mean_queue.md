# Hangzhou realworld

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 54.21 | 59.19 | 53.66 | **52.78** | 60.13 |
| MA2C | 48.70 | **45.41** | 48.18 | 48.82 | 48.37 |
| IQLL | 182.72 | 69.75 | 40.47 | **30.59** | 108.14 |
| PPO | 33.33 | 44.97 | 17.91 | 22.45 | **16.44** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.

Bold: lowest unrounded mean within each controller; exact ties included.
