# Grid | Test | Temporal

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 110.01 | 101.35 | **101.02** | 107.70 | 116.09 |
| MA2C | 87.05 | **83.64** | 88.62 | 90.62 | 90.83 |
| IQLL | 247.61 | **149.94** | 248.78 | 190.05 | 235.02 |
| PPO | 96.13 | 64.04 | 34.32 | 44.45 | **29.65** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
