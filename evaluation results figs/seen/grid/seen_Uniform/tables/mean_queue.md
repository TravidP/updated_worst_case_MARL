# Grid | Uniform demand

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 58.05 | 58.29 | **55.33** | 57.75 | 62.71 |
| MA2C | 52.82 | 51.51 | **50.59** | 51.26 | 50.71 |
| IQLL | 71.11 | 76.89 | 61.97 | **40.46** | 122.24 |
| PPO | 44.94 | 38.54 | 21.16 | 24.34 | **18.95** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen in-distribution evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
