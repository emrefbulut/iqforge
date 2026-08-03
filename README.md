# sigkit

SDR ile yakalanmış ham RF kayıtlarını (SigMF) makine öğrenmesinde kullanılabilir
etiketli veri setlerine çeviren komut satırı aracı.

> Geliştirme aşamasında — Faz 1 (iskelet + SigMF okuma + `info`) tamamlandı.
> Ayrıntılı README Faz 5'te yazılacak; yol haritası için [SPEC.md](SPEC.md).

## Hızlı deneme

```bash
uv sync --group dev
uv run sigkit info examples/sample.sigmf-meta
```

`examples/sample.sigmf-meta` paketle gelen sentetik kayıttır; donanım gerektirmez.
İçinde iki modülasyonlu burst (BPSK, QPSK) ve merkez frekanstan tam **+100 kHz**
kaymış sürekli bir referans ton bulunur. Referans ton ayrı bir annotation
(`ref_tone`) olarak işaretlidir ve sonraki fazlardaki doğrulamaların dayanağıdır.

## Lisans

MIT (Faz 5'te eklenecek).
