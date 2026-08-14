| stride | overlap | windows/rec | recording-level | window-level | inflation (paired) | n |
|---|---|---|---|---|---|---|
| 1024 | 0% | 14 | 66.8% ± 6.7% | 68.3% ± 2.1% | **+1.5 pp** ± 1.7 | 15 |
| 512 | 50% | 28 | 65.8% ± 9.0% | 69.4% ± 3.2% | **+3.7 pp** ± 2.3 | 15 |
| 128 | 88% | 112 | 65.1% ± 12.6% | 77.2% ± 4.3% | **+17.7 pp** ± 7.2 | 4 |

LoRaIQ, class = propagation environment (drone_los, drone_nlos, pedestrian_partial_los, pedestrian_nlos, indoor), 312 recordings over 13 capture sessions, grouped by transmission id so simultaneous receptions stay in one split. Window fixed at 1024 samples over a 15244-sample segment centred on each frame; no noise added. Overlap is the only thing that moves between rows. Inflation is the mean paired difference ± its standard error.
