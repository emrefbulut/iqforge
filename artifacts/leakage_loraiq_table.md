| stride | overlap | windows/rec | recording-level | window-level | inflation (paired) | n |
|---|---|---|---|---|---|---|
| 1024 | 0% | 14 | 66.8% ± 6.7% | 68.3% ± 2.1% | **+1.5 pp** ± 1.7 | 15 |
| 512 | 50% | 28 | 65.0% ± 1.0% | 71.8% ± 1.1% | **+6.8 pp** ± 1.1 | 3 |

LoRaIQ, class = propagation environment (drone_los, drone_nlos, pedestrian_partial_los, pedestrian_nlos, indoor), 312 recordings over 13 capture sessions, grouped by transmission id so simultaneous receptions stay in one split. Window fixed at 1024 samples over a 15244-sample segment centred on each frame; no noise added. Overlap is the only thing that moves between rows. Inflation is the mean paired difference ± its standard error.
