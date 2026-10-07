# Grid | Mixture scenario 2

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 76.28 | 75.66 | **70.98** | 75.50 | 81.74 |
| MA2C | 68.94 | 66.60 | **64.14** | 65.54 | 64.84 |
| IQLL | **62.07** | 78.51 | 64.28 | 91.48 | 65.11 |
| PPO | 53.03 | 51.92 | 24.58 | 29.12 | **22.43** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
