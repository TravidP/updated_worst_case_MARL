# Monaco | Test | Redistribution

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 144.05 | 221.76 | **121.79** | 166.55 | 144.16 |
| MA2C | 16.99 | **15.91** | 16.94 | 19.34 | 18.95 |
| IQLL | 334.11 | **231.57** | 310.16 | 314.89 | 308.28 |
| PPO | 145.68 | 226.11 | 10.23 | 7.91 | **7.58** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
