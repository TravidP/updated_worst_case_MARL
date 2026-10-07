# Grid | Test | Peak

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 66.84 | 68.60 | **62.90** | 66.06 | 73.98 |
| MA2C | 59.64 | 57.63 | **56.94** | 57.70 | 57.19 |
| IQLL | 75.51 | 111.73 | 68.71 | **45.28** | 136.93 |
| PPO | 49.21 | 45.05 | 23.71 | 27.62 | **21.49** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
