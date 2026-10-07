# Monaco | Test summary

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | 103.30 | 166.83 | **96.82** | 115.83 | 107.59 |
| MA2C | 21.93 | **20.75** | 24.23 | 31.69 | 29.90 |
| IQLL | 285.80 | **184.49** | 266.27 | 277.60 | 275.88 |
| PPO | 117.34 | 178.35 | 19.83 | 28.29 | **13.18** |

Equal weight per scenario; 12 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
