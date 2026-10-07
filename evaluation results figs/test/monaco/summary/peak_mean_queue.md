# Monaco | Test | Peak

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **178.98** | 242.03 | 181.44 | 202.74 | 187.87 |
| MA2C | 32.00 | **28.16** | 41.64 | 67.77 | 62.57 |
| IQLL | 339.38 | **228.81** | 319.33 | 319.49 | 314.83 |
| PPO | 190.40 | 250.21 | 29.01 | 42.13 | **21.06** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
