# Grid | Test | Redistribution

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 57.38 | 57.74 | **55.08** | 57.43 | 62.93 |
| MA2C | 52.10 | 50.58 | **50.21** | 50.66 | 50.38 |
| IQLL | 63.66 | 73.79 | 44.10 | **36.18** | 109.31 |
| PPO | 43.02 | 38.59 | 20.75 | 24.18 | **18.64** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
