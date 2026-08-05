# Phase 4 - seed grid results

- source: `examples`  ·  epochs: 20  ·  batch: 64
- `--balance-by core:freq_lower_edge`  ·  model: 13490 parameters

## Test accuracy

| split seed | train 0 | train 1 | train 2 | mean | std |
|---|---|---|---|---|---|
| 11 | 95.00% | 100.00% | 100.00% | **98.33%** | 2.89% |
| 22 | 100.00% | 100.00% | 100.00% | **100.00%** | 0.00% |
| 33 | 97.50% | 100.00% | 98.75% | **98.75%** | 1.25% |
| 44 | 100.00% | 100.00% | 93.75% | **97.92%** | 3.61% |
| 55 | 100.00% | 100.00% | 91.25% | **97.08%** | 5.05% |

## Spread

| source | value |
|---|---|
| all runs | 98.42% ± 2.81% (min 91.25%, max 100.00%) |
| across split seeds | 98.42% ± 1.08% |
| across training seeds (fixed split) | mean std 2.56%, largest 5.05% |
| dominant source | the TRAINING seed |
