# Monaco | Rapid switching every 900 s

Mean queue (vehicles; lower is better)

| Controller | Baseline | Random grouping | Domain randomization | Fixed WCE | Online WCE |
|---|---:|---:|---:|---:|---:|
| IA2C | **60.82** | 118.70 | 74.18 | 83.36 | 101.45 |
| MA2C | 27.93 | **26.06** | 31.33 | 32.47 | 30.51 |
| IQLL | 257.28 | **128.18** | 194.58 | 233.29 | 230.83 |
| PPO | 65.34 | 123.04 | 46.45 | 127.44 | **23.19** |

n = 10 per method; training seed = 101; table statistics are unsmoothed.
Frozen test evaluation.

Bold: lowest unrounded mean within each controller; exact ties included.
