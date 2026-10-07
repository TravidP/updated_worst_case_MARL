# Monaco real world data

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 139.59 | 151.30 | 165.33 | 147.30 | **126.15** |
| MA2C | 83.66 | **63.58** | 80.73 | 108.21 | 112.48 |
| IQLL | 217.98 | **214.64** | 218.25 | 241.50 | 216.97 |
| PPO | 89.00 | 115.08 | **23.13** | 46.08 | 26.88 |

n = 10 per method; training seed = 101; table statistics are unsmoothed.

Bold: lowest unrounded mean within each controller; exact ties included.
