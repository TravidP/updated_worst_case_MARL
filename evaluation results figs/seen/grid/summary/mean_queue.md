# Grid | Seen summary

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 100.09 | **89.17** | 107.49 | 115.22 | 117.12 |
| MA2C | 92.35 | **90.24** | 100.66 | 103.65 | 103.74 |
| IQLL | 242.75 | **164.92** | 190.13 | 174.14 | 244.21 |
| PPO | 92.81 | 60.89 | 44.52 | 58.33 | **41.86** |

Equal weight per scenario; 11 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
