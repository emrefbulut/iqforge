# TEKNOFEST uyumu

`iqforge`’un TEKNOFEST yarışmalarına nasıl oturduğu. Kategoriler yıldan yıla değişir; bu metin **2026** listesine (52 yarışma) göredir. 2027 isimleri kayabilir — başvurudan önce şartnameyi yeniden okuyun.

**Eşleme kuralı:** iqforge asla başvurunun kendisi değildir. Başvuru olan sistemin altındaki veri seti ve ölçüm katmanıdır. Canlı yakalama, yön bulma, karıştırma, demodülasyon ve arayüz `iqforge` kapsamı dışındadır (SPEC §2).

TEKNOFEST 2026 başvuruları 20–28 Şubat 2026’da kapandı. Bunu **takım kurmak ve 2027 hazırlığı** için kullanın; 2026’ya geç başvuru için değil.

---

## Tablo nasıl okunur

| Uyum | Anlam |
|---|---|
| **Çekirdek** | Puanlanan bir RF/ML görevi; etiketli IQ ve dürüst ayrım gerekir |
| **Destek** | Rapora veya bir alt sisteme yardım eder; yarışma asıl olarak donanım veya görüntüdür |
| **Yok** | Zorlamayın |

---

## Çekirdek — takım olarak girin, veri yolunda iqforge kullanın

### Elektronik Harp — ilk kez 2026

İki görev tipi. Ödül sıralaması asgari **hem ED hem ET** ister (tespit + parametre çıkarımı + yön bulma, artı bir karıştırma *ve* bir aldatma tekniği). Büyük ödül tüm görev grafiğini ister. iqforge yön bulma, karıştırma veya aldatma yapmaz.

| Şartname görevi | iqforge | Eksik |
|---|---|---|
| ED — tespit | Hayır. Enerji / CFAR / SDR zinciriniz | Donanım + DSP |
| ED — **parametre çıkarımı** | **Evet, ML yarısı.** Şartname taşıyıcı, bant genişliği, güç, analog/sayısal, **modülasyon (tercihen)**, protokol, çoklama, FHSS/DSSS ister. Ek parametre ek puan. Modülasyon/protokol sınıflandırıcıları eğitilmiş modellerdir; veri setleri iqforge’un işidir | Frekans/BW/güç için klasik kestiriciler; BPSK/QPSK’ten fazla sınıf |
| ED — dinleme / demod | Kapsam dışı | Demod yığını |
| ED — yön bulma / konum | Kapsam dışı | Dizi anten |
| ET — karıştırma / aldatma / GNSS aldatma | Kapsam dışı (gönderim) | Verici |
| **En iyi yapay zekâ uygulamaları ödülü** | **Tek ödül olarak en güçlü eşleşme.** Haberleşme ED’de YZ’nin özgünlük, hız, başarı, işlevselliğine verilir. Savunulabilir iddia: *doğruluk kayıt seviyesinde ayrımdan sonra ölçüldü; pencere seviyesindeki şişme gizlenmedi, raporlandı* | Sadece CLI değil, sahada sınıflandırıcının canlı gösterimi |

**Etrafında ne kurulur:** bir SDR + iqforge veri setleriyle eğitilmiş bir sınıflandırıcı (modülasyon / analog-sayısal) + raporda `audit`. Yön bulma ve ET için ortak.

### Çelikkubbe Hava Savunma Sistemleri

Yarışma sistemler sistemidir (algıla, tanı, izle, angaje ol). Ulusal mimaride görünen RF parçaları: ESM, drone RF kimliği, IFF’e yakın sınıflandırma. iqforge **anti-drone / yayıcı kimliği alt sistemine** oturur, komuta-kontrol resmine değil.

**AirID tipi** yapı zaten işlendi (bir kaydın patlama dilimleri ayrılmamalı). `--group-by` tam bu veriye uyar.

Yılın şartnamesinde açık bir RF sınıflandırma puanı yoksa **çekirdek değil, destek**.

### Güvenli Uydu Haberleşmesi / Hareketli Uydu Terminali

2026 hareketli terminal ağırlıklı olarak yönelme, stabilizasyon ve hareket altında bağlantıdır. Güvenli uydu haberleşmesi geçmişte karıştırmaya/aldatmaya dayanıklılığı da kapsar.

iqforge uyumu **dar ve gerçek:** SATCOM IQ üzerinde ML (dalga biçimi, girişim sınıfı) ve **dağılım kaymasına “sızıntı” dememek.** Methodology §6.2’deki Vega-C bu ders: farklı geçiş, Doppler, elevasyon, SNR; `audit` bunu artık LEAK değil RISK yazar.

SATCOM takımının **raporunda** veya şartname girişim sınıflandırmasını puanlıyorsa veri aracı olarak kullanın. Hareketli terminale bir CLI ile girmeyin.

### Üniversite Öğrencileri Araştırma Projeleri

iqforge’un *kendisinin* proje olduğu en yakın yol: RF-ML’de ayrım sızıntısı üzerine ölçüm, kamuya açık SigMF + açık kod. JOSS (yazılım) ile konferans (sonuç) aynı deneyden çıkan iki ayrı üründür.

---

## Destek — takımda zaten RF veya etiketli sinyal sorunu varsa

| Yarışma | IQ’ya neden değebilir | iqforge ne yapar | Ne yapmaz |
|---|---|---|---|
| FPV drone izleme (2026 yeni) | Bazı yıllarda görüntü/RF karma | RF varlık veya kimlik başı için veri seti | Kutu, video |
| Havacılıkta yapay zekâ | Genelde görüntü / sim / eniyileme | Yalnızca o yılın görevi radyo veya transponder IQ içeriyorsa | Tipik görüntü hattı |
| 5G ve YZ ile akıllı yol güvenliği | Şartname havadan IQ kullanıyorsa 5G / C-V2X | SigMF’ten doluluk / girişimci sınıfları | Yol sahnesi ML |
| Model uydu / roket / dikey iniş | Telemetri RF, yer istasyonu | Etiketli telemetri patlamaları, dürüst ayrım | Uçuş mekaniği, GNC |
| Sürü İHA / savaşan İHA / drone şampiyonası | C2 linkleri, dost-düşman RF, GNSS aldatma tespiti | Sınıflandırıcı veri setleri; rapor için `audit` | Gövde, otonomi |
| İnsansız deniz / su altı | Akustik SigMF-IQ değil; aracın RF haberleşmesi olabilir | Yalnızca kaydedilmiş RF komut linki | Sonar ML |
| Çip tasarım | Çip RF ön uçsa ve IQ toplanıyorsa | Silikon sonrası IQ → veri seti | RTL, PDK |
| Kuantum (yazılım) | Düşük olasılık | Radyo deneyi yoksa yok | Kuantum yığını |
| Maden / sanayide robotik | RF salınımından kestirimci bakım zorlama | Endüstriyel EMI sınıflandırması gibi | Mekanik robot |

---

## Yok — eşlemeye zaman ayırmayın

Blokzincir, e-ticaret, fintech, doğal dil işleme, YZ film, mimari, biyoteknoloji, onkoloji, lise iklim/kutup araştırması, hyperloop, nükleer *tasarım*, jet motor *tasarım*, uçan araba simülasyonu, lojistik eniyileme, PARDUS hata avı, Robolig, TravelX, mesleki yetenek, insanlık yararına (K-12), elektrikli araç yarışları, Robotaksi (radyo alt modülü puanlanmıyorsa).

Bunlara iqforge zorlamak kimliği sulandırır ve puan getirmez.

---

## Elektronik harp — görev grafiği (ayrıntı)

```
Spektrum
    → tespit          [DSP / SDR’niz]
    → parametre çıkar [klasik + ML]
         ML sınıfları ← iqforge build / Dataset / audit
    → dinle/demod     [kapsam dışı]
    → yön bul         [kapsam dışı]
    → ET ata          [kapsam dışı]
```

**ED parametre çıkarımında işe yarar bir takım arkadaşı olmak için asgari**

1. Sahanın yayınlayacağı ailelerin donanım kaydı (2026 şartnamesine göre amatör telsiz, ISM modülleri), SigMF olarak.
2. `{bpsk, qpsk}`’ten geniş etiketler: analog/sayısal, birkaç sayısal modülasyon, isteğe bağlı FHSS / sabit.
3. Kayıt (veya `--group-by`) seviyesinde `iqforge build`; teknik rapora `audit`.
4. Küçük bir sınıflandırıcı (`train` duman testidir, yarışma modeli değil). Gerçek mimariyi siz koyun; veri sözleşmesini koruyun.

Demo sayısını “şişirmek” için CLI’ya gizli pencere seviyesi ayrımı **koymayın**. Araç tam olarak o bozulmayı önlemek için var. Kötü ayrımı rapordaki tablo için `scripts/` altındaki bir ölçüm script’i çalıştırabilir.

---

## Sizin için dürüst sıra (tek kişi, bu repo)

1. **Araştırma projesi / bildiri** — en yüksek kaldıraç; ROADMAP Now ile örtüşür (sızıntıyı gerçek kayıtlarda tekrarlamak).
2. **EH takımı, YZ ödülü + parametre çıkarımı puanı** — yüksek kaldıraç; ortak ve radyo ister.
3. **Anti-drone / Çelikkubbe alt sistemi** — 2027 şartnamesi RF kimliği puanlıyorsa.
4. **SATCOM rapor eki** — Vega-C dersi; az zaman.
5. Geri kalanı — yalnızca takım arkadaşının zaten aracı varsa ve radyo ML yan kolu istiyorsa.

Eksik olan yeni bir yarışma fikri değil. **Radyosu olan üç kişi** (ROADMAP Next) ve kamuya açık gerçek SigMF üzerinde bir sızıntı tablosu.
