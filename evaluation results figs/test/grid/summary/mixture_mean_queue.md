# Grid | Test | Mixture

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 86.38 | 90.62 | **84.80** | 95.80 | 110.74 |
| MA2C | 79.26 | 76.42 | **75.26** | 79.70 | 79.93 |
| IQLL | **83.15** | 89.54 | 86.92 | 152.87 | 113.62 |
| PPO | 63.58 | 55.97 | 26.13 | 32.64 | **24.05** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
