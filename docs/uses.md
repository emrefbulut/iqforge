# iqforge ne işe yarar

Savunma laboratuvarı, üniversite grubu veya işe alım için tek sayfa. Özellik listesi değil.

## İddia

Elinizde SDR kayıtları var. Eğitilmiş bir model istiyorsunuz. Bu iki cümle arasında yanlış yapılması kolay, fark edilmesi zor kararlar durur. Sonuçları sessizce bozanı train/test ayrımıdır.

Pencere seviyesinde bölme, komşu pencereleri — aynı semboller, aynı gürültü, aynı sönümleme, yarım adım kaymış — hem eğitime hem teste koyar. Raporlanan doğruluk **yükselir**. Model kötüleşir. Sayıya bakınca hiçbir şey şüpheli görünmez.

Kontrollü iki sınıflı bir görevde bu şişme tepe noktada **+13,6 yüzde puan** oldu (eşleştirilmiş, 15 tohum çifti, patlama SNR −2,2 dB). Yüksek SNR’de her iki kol da tavana oturur; sızıntı görünmez. Bkz. [methodology.md](methodology.md) §2.

`iqforge` **kayıt** seviyesinde böler. Geçerli katmanlı bir ayrım mümkün değilse geri düşmez, **hata verir**. Ürün bu reddediştir.

## Ne değildir

Sınıflandırıcı, detektör, demodülatör veya elektronik harp sistemi değildir. Onların altındaki katmandır: mevcut SigMF kayıtlarını, bir incelemede savunabileceğiniz etiketli bir PyTorch veri setine çevirir.

## Bu sayı nerede işe yarar

| Ortam | Ne eğitilir | Sızıntı nasıl görünür |
|---|---|---|
| Elektronik destek / COMINT | Modülasyon, protokol, yayıcı kimliği | Aynı oturum eğitim ve testte; model sınıfı değil kaydı ezberler |
| Özgül yayıcı tanımlama | “Hangi tip değil, hangi radyo” | Aynı yayıcının komşu pencereleri kimliği sızdırır |
| Anti-drone / RF parmak izi | İniş yolundan drone tipi | Aynı uçuşun patlama dilimleri bağımsız örnek sayılır (AirID tipi veri) |
| Uydu haberleşme / telemetri ML | Doppler altında dalga biçimi veya uydu | Farklı geçişler farklı koşul gibi durur; buna “sızıntı” demek yanlış teşhistir (Vega-C: dağılım kayması, örtüşme değil) |
| Spektrum izleme / regülatör | Doluluk, kaçak verici | Aynı yer, aynı saat, iki dosya — yapıca ayrı, fiziken aynı kanal |
| Akademik RF-ML makaleleri | Herhangi bir IQ CNN | Pencere karıştırma hâlâ varsayılan; şişme görevin *sınırda* olduğu yerde en büyük — ki makaleler de orada “atılım” iddia eder |

`iqforge audit` neyi kontrol ettiğini, ne bulduğunu ve neyi kontrol edemediğini yazar. İncelenmemiş alanı gizleyen bir “temiz” durumu yoktur. Örtüşen yayın süresi, aynı baytlar, iki kutuya düşen bir kayıt ve her ayrıma *farklı* bir geçiş koyan tekrarlı zaman damgası ayrı hatalar olarak adlandırılır — bir skoru zıt yönlerde hareket ettirirler.

## Bugün ne yapılabilir

```
iqforge info    capture.sigmf-meta
iqforge inspect capture.sigmf-meta
iqforge build   recordings/ -o dataset/ --balance-by core:freq_lower_edge
iqforge stats   dataset/
iqforge audit   dataset/
```

`cf32_le`, `ci16_le`, `ci8` okur. Tamsayı yolları kamuya açık kayıtlara karşı kontrol edildi (tam ölçek, I/Q sırası) — sizin radyoya karşı değil. `--group-by` ilgili dosyaları bir arada tutar (yol regex, CSV veya `core:collection`). PyTorch isteğe bağlıdır.

## Neyin yerini tutmaz

Donanım. Yön bulma dizisi. Karıştırıcı. Makale. Tedarik test planı. Bunlar başka sistemlerdir. iqforge, o sistemlerin sızmış bir ayrımdan gelmiş bir sayıyı alıntılamasını durdurur.

## Atıf

[github.com/emrefbulut/iqforge](https://github.com/emrefbulut/iqforge) — MIT. Ölçüm tabloları `artifacts/` içinde; koşullar (cihaz, kütüphane sürümleri, n, aralık) sayıyla birlikte gelir.
