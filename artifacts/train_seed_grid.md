# Faz 4 — tohum ızgarası sonuçları

- kaynak: `examples`  ·  epoch: 20  ·  batch: 64
- `--balance-by core:freq_lower_edge`  ·  model: 13490 parametre

## Test doğruluğu

| bölme tohumu | eğitim 0 | eğitim 1 | eğitim 2 | ortalama | std |
|---|---|---|---|---|---|
| 11 | 50.00% | 85.00% | 50.00% | **61.67%** | 20.21% |
| 22 | 61.88% | 50.00% | 50.00% | **53.96%** | 6.86% |
| 33 | 50.62% | 50.00% | 50.00% | **50.21%** | 0.36% |
| 44 | 50.62% | 50.00% | 50.00% | **50.21%** | 0.36% |
| 55 | 50.00% | 50.00% | 50.00% | **50.00%** | 0.00% |

## Saçılma

| kaynak | değer |
|---|---|
| tüm koşular | 53.21% ± 9.31% (min 50.00%, maks 85.00%) |
| bölme tohumları arası | 53.21% ± 5.01% |
| eğitim tohumları arası (sabit bölmede) | ortalama std 5.56%, en büyük 20.21% |
| baskın kaynak | EĞİTİM tohumu |
