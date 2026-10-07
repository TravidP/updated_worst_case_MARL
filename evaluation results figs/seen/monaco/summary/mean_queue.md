# Monaco | Seen summary

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **52.74** | 69.95 | 57.02 | 62.02 | 56.16 |
| MA2C | **22.14** | 22.28 | 23.91 | 28.74 | 24.86 |
| IQLL | 155.07 | **94.89** | 128.35 | 143.21 | 133.86 |
| PPO | 52.67 | 68.95 | 24.61 | 31.68 | **20.72** |

Equal weight per scenario; 11 scenarios; n = 10 per scenario/method; training seed = 101.
Unsmoothed Mean queue; best values compared within controller.

Bold: lowest unrounded mean within each controller; exact ties included.
