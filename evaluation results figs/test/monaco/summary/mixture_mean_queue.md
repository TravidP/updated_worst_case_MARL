# Monaco | Test | Mixture

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 37.97 | 104.76 | **28.85** | 30.72 | 32.46 |
| MA2C | 14.99 | 15.94 | 14.29 | 14.56 | **14.12** |
| IQLL | 224.66 | **141.89** | 221.58 | 238.09 | 243.37 |
| PPO | 73.46 | 118.69 | 10.56 | 8.15 | **7.65** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
