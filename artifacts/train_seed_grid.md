# Faz 4 — tohum ızgarası sonuçları

- kaynak: `examples`  ·  epoch: 20  ·  batch: 64
- `--balance-by core:freq_lower_edge`  ·  model: 13490 parametre

## Test doğruluğu

| bölme tohumu | eğitim 0 | eğitim 1 | eğitim 2 | ortalama | std |
|---|---|---|---|---|---|
| 11 | 95.00% | 100.00% | 100.00% | **98.33%** | 2.89% |
| 22 | 100.00% | 100.00% | 100.00% | **100.00%** | 0.00% |
| 33 | 97.50% | 100.00% | 98.75% | **98.75%** | 1.25% |
| 44 | 100.00% | 100.00% | 93.75% | **97.92%** | 3.61% |
| 55 | 100.00% | 100.00% | 91.25% | **97.08%** | 5.05% |

## Saçılma

| kaynak | değer |
|---|---|
| tüm koşular | 98.42% ± 2.81% (min 91.25%, maks 100.00%) |
| bölme tohumları arası | 98.42% ± 1.08% |
| eğitim tohumları arası (sabit bölmede) | ortalama std 2.56%, en büyük 5.05% |
| baskın kaynak | EĞİTİM tohumu |
