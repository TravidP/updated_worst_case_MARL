# Grid | Test summary

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 80.15 | 79.58 | **75.95** | 81.75 | 90.94 |
| MA2C | 69.51 | **67.07** | 67.76 | 69.67 | 69.58 |
| IQLL | 117.48 | 106.25 | 112.13 | **106.10** | 148.72 |
| PPO | 62.98 | 50.91 | 26.23 | 32.22 | **23.46** |

Equal weight per scenario; 12 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
