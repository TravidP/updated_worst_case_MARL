# Monaco | Test | Temporal

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **52.20** | 98.76 | 55.18 | 63.32 | 65.89 |
| MA2C | 23.73 | **22.99** | 24.06 | 25.10 | 23.96 |
| IQLL | 245.03 | **135.68** | 214.00 | 237.95 | 237.03 |
| PPO | 59.83 | 118.37 | 29.52 | 54.99 | **16.44** |

Equal weight per scenario; 3 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
