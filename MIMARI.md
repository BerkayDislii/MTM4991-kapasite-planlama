# MIMARI.md — Kubernetes Ortamında Kaynak Kullanımı Tahmini ve Tamsayılı Doğrusal Programlama ile Maliyet-Optimal Kapasite Planlaması

**Proje:** MTM4991 Çok Disiplinli Tasarım Projesi (YTÜ Matematik Mühendisliği)
**Bu dosyanın statüsü:** Bağlayıcı mimari spesifikasyon. Aşağıdaki kararlar proje sahibi tarafından onaylanmıştır.

---

## 0. Ajan İçin Çalışma Kuralları (ÖNCE OKU)

1. **Bu dosyadaki kararları değiştirme.** Bir karar veriyle veya teknik gerçeklikle çelişiyorsa DUR, kanıtı (komut çıktısı, hata, veri istatistiği) göster ve kullanıcıya sor. Sessizce alternatif seçme.
2. **Uydurma yok.** Sonuç sayısı, fiyat, kaynakça, veri şeması veya sütun adı uydurma. Her sayı kodun ürettiği çıktıdan gelmeli. Bir bilgi bu dosyada "DOĞRULANACAK" etiketiyle işaretliyse gerçek dosyadan/komuttan doğrula (`assert` ile) ve kullanıcıya bildir.
3. **Parametreleri koda gömme.** Tüm sayısal parametreler `configs/*.yaml` içinde olacak. "VARSAYILAN" etiketli değerler başlangıç değeridir, kullanıcı değiştirebilir.
4. **Veri sızıntısı yasak.** Test dönemi eğitim, ölçekleme, hiperparametre seçimi veya model seçiminde kullanılamaz. Ölçekleyiciler yalnızca eğitim verisine uydurulur.
5. **Kod kalitesi:** Açıklayıcı **Türkçe yorum satırları**, tip ipuçları (type hints), kısa docstring'ler. Büyük veride satır satır Python döngüsü yok (vektörel pandas/numpy/polars/pyarrow). Rastgelelik tohumla sabitlenir.
6. **Lisans gerektiren yazılım yasak** (Gurobi, CPLEX vb.). ILP yalnızca Google OR-Tools ile çözülür.
7. **Büyük indirmeleri yapma.** Alibaba verisi anket arkasında olduğu için kullanıcı `data/raw/` altına elle koyar. `container_usage` (yaklaşık 28 GB sıkıştırılmış) **kullanılmayacak ve indirilmeyecek.**
8. **Küçük adımlarla ilerle.** Her kilometre taşı (Bölüm 9) için kod + test + kısa özet ver, kullanıcı onayı olmadan sonraki taşa geçme.
9. **Gerçek veri ile sentetik senaryo sonuçları asla karıştırılmaz** (ayrı klasör, ayrı tablo).

---

## 1. Teknoloji Yığını

| Katman | Araç | Not |
|---|---|---|
| Dil / ortam | Python 3.10–3.12, `venv` | Linux (yerel). Kurulum sonrası sürümler `requirements.txt`'ye `pip freeze` ile yazılır. |
| Veri | `pandas`, `numpy`, `pyarrow` (Parquet); büyük dosyalar için `polars` | Alibaba `machine_usage` (~1,7 GB sıkıştırılmış) parça parça okunur. |
| Klasik tahmin | `statsmodels` (SARIMAX) | Mevsim uzunluğu 24 saat. |
| Makine öğrenmesi | `PyTorch` (CUDA), `XGBoost` (GPU) | GPU: RTX 3050. VRAM varyanta göre değişir (4/6/8 GB): `nvidia-smi` ile oku, batch size'ı buna göre ayarla, `torch.autocast` (AMP) kullan. XGBoost için `device="cuda"`, `tree_method="hist"`. |
| Optimizasyon | `ortools` (Google OR-Tools) | Birincil: `pywraplp` ile **SCIP** (MIP). Doğrulama için küçük örneklerde **CP-SAT** (tamsayı ölçekleme ile). Kurulumda `SCIP` backend'inin mevcut olduğunu smoke test ile doğrula. |
| Değerlendirme / yardımcı | `scikit-learn` (ölçekleyici, metrik), `scipy`, `matplotlib`, `pyyaml`, `tqdm`, `joblib`, `pytest` | Yeni bağımlılık eklemeden önce kullanıcıya sor. |
| Şekiller | `matplotlib` | Mümkünse serif yazı tipi (Times New Roman / Liberation Serif). Türkçe karakterler doğru görünmeli. Şekil başlığı görselin içine yazılmaz (raporda altyazı olarak girilecek). |

**Donanım/ortam doğrulaması (M0'da zorunlu):** `torch.cuda.is_available()`, `nvidia-smi` çıktısı, OR-Tools SCIP smoke test, statsmodels SARIMAX küçük örnek çalıştırma.

---

## 2. Klasör Yapısı

```
proje/
├── MIMARI.md
├── requirements.txt                 # (ek) kurulumdan sonra üretilir
├── data/                            # git'e girmez
│   ├── raw/
│   │   ├── bitbrains/               # fastStorage/ ve Rnd/ alt klasörleri (VM başına CSV)
│   │   └── alibaba/                 # machine_meta, machine_usage, container_meta
│   ├── interim/                     # VM/makine bazında 5 dk ızgarasına hizalı Parquet
│   └── processed/                   # pool_<ad>.parquet (havuz toplam talebi)
├── configs/
│   ├── pools.yaml                   # havuz tanımları (tohum, boyut, üyeler)
│   ├── server_types.yaml            # AWS sunucu tipleri ve maliyetler (tarihli anlık görüntü)
│   ├── preprocess.yaml              # (ek) boşluk limiti, M_ref, bölme günleri
│   ├── forecast.yaml                # (ek) model hiperparametreleri, ufuk, tohumlar
│   ├── ilp.yaml                     # (ek) alpha, s_k, d_k, L_k, çözücü ayarları
│   ├── synthetic.yaml               # (ek, M7) sentetik senaryo tanımları ve tohum
│   └── rapor.yaml                   # (ek, M8) şekil/tablo kataloğu, numaralandırma sırası, altyazı önerileri
├── src/
│   ├── common/                      # (ek) io, tohum, metrik, günlükleme
│   ├── ingest.py                    # ham veriyi okur, 5 dk ızgarasına hizalar
│   ├── preprocess.py                # birim dönüşümü, havuz toplamı, saatlik tepe
│   ├── eda.py                       # keşifsel analiz ve şekiller
│   ├── forecast/                    # seasonal_naive, sarima, xgb, lstm_gru, (opsiyonel) transformer
│   ├── ilp/                         # model.py (çok dönemli ILP), solve.py
│   ├── baselines/                   # (ek) tepkisel yöntemler
│   └── evaluate.py                  # (ek) maliyet ve yetersizlik ölçütleri
├── tests/                           # (ek) birim testleri
└── outputs/                         # (ek)
    ├── eda/                         # şekiller ve tablolar
    ├── forecasts/<dataset>/<pool>/<model>/
    ├── models/                      # (ek, M4) tahmin ölçüt tabloları, SARIMA seçimi, LSTM günlükleri
    ├── plans/<dataset>/<pool>/<strategy>/
    ├── ilp/                         # (ek, M5, kullanıcı talebi) ILP planları (ilp/plans/<dataset>/<pool>/<model>/) ve özet tablolar
    ├── reactive/                    # (ek, M6, kullanıcı talebi) R0/R1/R2 simülasyonları (reactive/sim/<dataset>/<pool>/) ve ILP ile karşılaştırma tabloları
    ├── results/                     # özet tablolar (CSV); results/izgara/ = M7 (α, w, L) ızgarası
    ├── synthetic/                   # sentetik senaryolar (gerçek sonuçlardan AYRI)
    └── rapor/                       # (ek, M8) sekil_<b>_<n>_*.png (+ _veri.csv), tablo_<b>_<n>_*.csv, katalog.csv/.md
```

---

## 3. Veri Setleri

| | Bitbrains GWA-T-12 (ANA) | Alibaba cluster-trace-v2018 (İKİNCİ) |
|---|---|---|
| Ölçek | fastStorage: 1.250 VM; Rnd: 500 VM | ~4000 makine, 8 gün |
| Örnekleme | ~5 dk | DOĞRULANACAK (şemada belirtilmemiş) |
| Süre | ~30 gün | 8 gün |
| Seviye | VM | Makine (çevrimiçi servis + toplu iş birlikte) |
| CPU | Mutlak (çekirdek sayısı, MHz) | `cpu_util_percent` [0,100] + `cpu_num` |
| Bellek | Mutlak (KB) | `mem_util_percent` [0,100]; fiziksel GB **yok** |
| Maliyet | Yok | Yok |

**Kullanılacak Alibaba dosyaları:** `machine_usage`, `machine_meta`, (opsiyonel, küçük) `container_meta`. `batch_*` ve `container_usage` kullanılmaz.

**DOĞRULANACAK (gerçek dosyada `assert` ile):**
- Bitbrains sütun adları (beklenen: `Timestamp [ms]`, `CPU cores`, `CPU capacity provisioned [MHZ]`, `CPU usage [MHZ]`, `Memory usage [KB]`). Adlar boşluk/şapka farkı içerebilir, `str.strip()` uygula ve eksik sütunda DUR.
- `Timestamp [ms]` birimi: ardışık farkların medyanı ≈ 300 ise **saniye**, ≈ 300000 ise milisaniye. Sütun adına güvenme, farkı ölç.
- Alibaba `machine_usage` örnekleme aralığı (ardışık `time_stamp` farkı), `cpu_num` makineler arası homojen mi, `machine_meta` zamanla değişiyor mu.
- Alibaba saat dilimi/mutlak tarih bilinmiyor: zaman `time_stamp` saniyesinden türetilen **göreli** dilim indeksi olarak ele alınır (gün sınırı = 24 saatlik bloklar).
- Bitbrains CSV'leri `;` ayraçlı, VM başına bir dosya. Rnd üç aylık (2013-7, 2013-8, 2013-9) alt klasöre bölünmüştür. fastStorage tek aydır (2013-8). 
- Ay klasörlerinin ayrı dönem mi yoksa birleştirilebilir mi olduğu M1'de veriden belirlenir, DURUP kullanıcıya sorulur; varsayım yapılmaz.
- Veri kaynağı atfı zorunlu: "Bitbrains IT Services Inc." (raporda belirtilecek).
---

## 4. Veri Ön İşleme Kuralları

### 4.1 Zaman hizalama
- Tüm seriler ortak **5 dakikalık ızgaraya** hizalanır (`floor`). Aynı ızgara hücresine düşen birden fazla ölçüm varsa ortalama alınır.

### 4.2 Birim dönüşümü
- **Bitbrains CPU (çekirdek eşdeğeri):**
  `cpu_cores = CPU_usage_MHz × CPU_cores / CPU_capacity_provisioned_MHz`
  (VM'in kendi tanımlı kapasitesine göre; dış frekans varsayımı yok.)
- **Bitbrains bellek:** `mem_gb = Memory_usage_KB / 1024²` (GiB).
- **Alibaba CPU:** `cpu_cores = cpu_util_percent / 100 × cpu_num`
- **Alibaba bellek (M_ref kuralı):** fiziksel bellek bilinmediği için
  `mem_gb = mem_util_percent / 100 × M_ref`, burada `M_ref = cpu_num × r` (makine başına referans bellek).
  - `r` (GB/çekirdek) bir **parametredir.** Merkez değer **4** (VARSAYILAN). **Duyarlılık analizi ızgarası: {2, 4, 8}** (AWS'deki c / m / r ailelerinin tipik bellek/çekirdek oranlarına karşılık gelir).
  - Alibaba bellek sonuçları raporda **"M_ref varsayımına bağlı"** olarak etiketlenir. Tüm Alibaba deneyleri 3 r değeri için de çalıştırılır.
- Geçersiz değerler (`-1`, `101`) → `NaN`.

### 4.3 Eksik veri ve VM yaşam süresi
- VM/makinenin ilk ve son gözlemi arasındaki kısa boşluklar (`max_gap_steps`, VARSAYILAN 3 adım = 15 dk) **doğrusal enterpolasyonla** doldurulur.
- İlk gözlemden önce ve son gözlemden sonra talep **0** sayılır (VM henüz yok/artık yok).
- Yaşam süresi içindeki **uzun** boşluklar enterpole edilmez, `NaN` kalır; havuz toplamında `skipna=True` ile 0 sayılır ve bu hücrelerin oranı `outputs/eda/quality_report.json` dosyasına yazılır. **Oran toplam hücrelerin %1'ini aşarsa DUR ve kullanıcıya bildir.**

### 4.4 Havuz toplamı ve saatlik TEPE (kritik kural)
Sıra **kesinlikle** şöyledir:
1. Her 5 dakikalık adımda **havuz toplamı** hesaplanır: `D5_t = Σ_v demand_v,t` (CPU ve bellek için ayrı).
2. Sonra saatlik **tepe** alınır: `D_h = max(D5_t : t ∈ saat h)` (12 adım).

> **Yasak:** VM/makine bazında saatlik tepe alıp bunları toplamak. Bu, eş zamanlı olmayan tepeleri toplayarak talebi şişirir. Tepe, **toplamın tepesidir.**

Ek olarak `D_h_mean` (saatlik ortalama) sadece raporlama/karşılaştırma için ayrı sütunda saklanır. Planlama talebi **tepedir.**

`data/processed/pool_<ad>.parquet` sütunları: `slot` (saatlik dilim indeksi), `cpu_peak`, `mem_peak`, `cpu_mean`, `mem_mean`, `n_active` (aktif VM/makine sayısı).

### 4.5 Havuz tasarımı (`configs/pools.yaml`)
- **Bitbrains:** `fastStorage` ve `Rnd` iki ayrı doğal havuz (**asla karıştırılmaz**). Ek olarak fastStorage'dan sabit tohumla rastgele ayrık alt havuzlar.
- **Alibaba:** sabit tohumla rastgele ayrık makine havuzları (VARSAYILAN: 500'er makine).
- Tüm sonuçlar havuzlar üzerinden **ortalama ± standart sapma** olarak raporlanır.

### 4.6 Zaman bölmesi
- Kronolojik, karıştırma yok. **Bitbrains:** ~20 / 4 / 6 gün (eğitim / doğrulama / test). **Alibaba:** 4 / 1 / 1 gün (96 / 24 / 24 saat), yalnızca 2–7. günler üzerinde.
- **Alibaba analiz penceresi (kullanıcı kararı, 2026-10-29, M3 sonrası "1" seçeneği):** İlk iki gün (0–1. günler, hücre < 576) atılır. Gerekçe: M3'te uzun boşluk oranı 8 günün tamamında %8,76 çıktı (eşik %1). İlk ~39 saatte makinelerin yalnızca %58–77'si ölçüm gönderiyor, 39–44. saatlerde küme genelinde kesinti var (40–43. saatlerde %0). 2–7. günlerde oran %0,15. Dilim 0 = 2. günün ilk saati. 4/1/1 bölmesi LSTM'in 48 saatlik bakış penceresine yeterli eğitim örneği bırakmak için seçildi.
- Alibaba'da 8 gün (kullanılabilir 6 gün) olduğundan haftalık mevsimsellik öğrenilemez ve günlük mevsimsellik yalnızca birkaç döngüyle sınırlıdır. Bu bir **bilinen sınırlılıktır**, raporlanacak.
- Test dönemi **ileri yürüyen (walk-forward)** değerlendirilir: her gün başında önceki tüm veriyle sonraki 24 saat tahmin edilir (Bölüm 5).

---

## 5. Tahmin ve Optimizasyon Boru Hattı

```
ham veri → ingest → preprocess → pool_<ad>.parquet (saatlik tepe)
                                       │
                         forecast/<model> (gün-ileri, 24 dilim)
                                       │   çıktı: D̂_t^cpu, D̂_t^mem
                                       ▼
                      ilp/ (çok dönemli ILP, Bölüm 6)  ──►  plan x_{k,t}
                                       │
                                       ▼
        evaluate.py: planı GERÇEK talep D_t üzerinde değerlendirir
```

**Temel kural: tahmin çıktısı ILP'ye doğrudan girdidir.** ILP gerçek talebi görmez. Plan `D̂`'ye göre yapılır, **başarı gerçekleşen `D` üzerinden ölçülür.**

### 5.1 Tahmin görevi
- **Hedef:** Havuz başına iki seri: `cpu_peak` (çekirdek), `mem_peak` (GB). Ufuk: **24 saatlik dilim (gün-ileri).**
- **Walk-forward:** Gün `d` için `d·24` öncesindeki veri kullanılır. Model parametreleri test boyunca **sabit** (eğitim+doğrulama ile bir kez eğitilir). SARIMA durumu `refit=False` ile yeni gözlemlerle güncellenir (API'yi doğrula). LSTM/GRU ağırlıkları sabit, girdi penceresi güncellenir.

### 5.2 Modeller (öncelik sırası)
| Öncelik | Model | Not |
|---|---|---|
| P0 | **Mevsimsel naif** (24 saat öncesinin değeri) | Her karşılaştırmanın tabanı. |
| P0 | **Gün-ileri son değer** (gün başındaki son gözlenen tepe, 24 dilim boyunca sabit) | Kullanıcı kararı (2026-10-29, M2 sonrası "A" seçeneği): mevsimsel naife ek taban. |
| P0 | **SARIMA** (`statsmodels` SARIMAX, mevsim=24) | Saatlik tepe serisi üzerinde. Parametre seçimi küçük bir ızgara + AIC (doğrulama dönemi kontrolüyle). ARIMA'nın `d` değeri durağanlık testlerine (ADF/KPSS) göre. Kullanıcı kararı (2026-10-29, "A"): ızgaraya **mevsimsel olmayan** (P=D=Q=0) adaylar da eklenir; mevsimsel/mevsimsel olmayan seçimi **AIC'ye** bırakılır. **Güncelleme (kullanıcı kararı, 2026-10-29, M4):** AIC yalnızca aynı $(d,D)$ ailesi içinde karşılaştırılır; mevsimsel ve mevsimsel olmayan aile kazananları arasındaki nihai seçim **doğrulama dönemi gün-ileri MAE'sine** göre yapılır. $D$: eğitim döneminde STL mevsimsellik gücü $F_s \ge 0{,}64$ ise 1. Kullanıcı M4 talebinde bu modeli "P1" olarak adlandırdı. |
| P1 | **LSTM** ve **GRU** (PyTorch, GPU) | Girdi: geçmiş `lookback_hours` (VARSAYILAN 48) saatin **5 dk çözünürlüklü** havuz toplamları (CPU, bellek kanalları). Çıktı: sonraki 24 saatlik tepe (CPU+bellek). Kayıp: MSE. Erken durdurma (doğrulama). AMP. **3 farklı tohum**, ortalama ± std. **Alibaba (kullanıcı talebi, 2026-10-29, M4):** bakış penceresi 24 saat (96 saatlik eğitimde 48 sa ile 25, 24 sa ile 49 eğitim kökeni). Son model eğitim+doğrulama ile, erken durdurmanın bulduğu epoch sayısı kadar yeniden eğitilir. Kullanıcı M4 talebinde LSTM'i "P2" olarak adlandırdı; GRU, XGBoost, DLinear bu M4 turunda çalıştırılmadı. Eğitim süresi/VRAM sorunu çıkarsa girdi çözünürlüğünü düşürmek bir **sapmadır** → önce kullanıcıya sor. |
| P2 | **XGBoost** (GPU) | Gecikme (lag) özellikleri + saat-sin/cos. Çok ufuklu: her ufuk için ayrı model veya `MultiOutput`. |
| P2 | **DLinear benzeri basit doğrusal model** | Derin modellerin basit modeli geçip geçmediğini görmek için (literatürdeki Zeng vd. uyarısı). |
| P3 | **Transformer** (opsiyonel) | Yalnızca P0–P2 tamamlandıktan sonra ve zaman kalırsa. |

### 5.3 Tahmin ölçütleri
MAE, RMSE, **MASE** (mevsimsel naif 24 saat ölçekli) ve **eksik tahmin oranı** (`D̂ < D` olan dilimlerin oranı). Hepsi test döneminde, CPU ve bellek için ayrı. **Ek (kullanıcı talebi, 2026-10-29, M4):** MAPE de hesaplanır; ölçütler eğitim, doğrulama ve test bölümleri için ayrı ayrı `outputs/models/` altına yazılır (eğitim/doğrulama seçim içindir, karşılaştırma testtedir).

### 5.4 Tahmin → ILP arayüz sözleşmesi
Tahminler `outputs/forecasts/<dataset>/<pool>/<model>/forecast_<gün>.parquet` dosyasına şu sütunlarla yazılır: `slot`, `cpu_hat`, `mem_hat`. (İleride kantil eklenirse `cpu_q90`, `mem_q90`.) ILP modülü **yalnızca bu dosyayı** okur.

---

## 6. ILP Modeli: Çok Dönemli, Unit Commitment'tan Esinlenen

Bu model, saatlik bağımsız bir "sırt çantası" **değildir.** Dönemler; sunucu açma/kapama maliyetleri ve minimum çalışma süresi kısıtlarıyla bağlanır. Mantık, elektrik sistemlerindeki devreye alma (unit commitment) problemindeki jeneratör davranışına benzer: bir sunucuyu açmanın maliyeti vardır ve açıldıktan sonra hemen kapatılamaz.

### 6.1 Küme, parametre ve karar değişkenleri

**Kümeler:** `k ∈ K` sunucu tipleri, `t ∈ {1,…,H}` zaman dilimleri (saat). Planlama ufku `H = 24` (VARSAYILAN).

| Sembol | Anlamı | Kaynak |
|---|---|---|
| $c_k$ | k tipi sunucunun saatlik maliyeti | `server_types.yaml` (AWS fiyat listesi, tarihli) |
| $C_k$ | k tipi sunucunun çekirdek kapasitesi | `server_types.yaml` |
| $M_k$ | k tipi sunucunun bellek kapasitesi (GB) | `server_types.yaml` |
| $s_k$ | k tipi bir sunucuyu **açma (başlatma) maliyeti** | `ilp.yaml` |
| $d_k$ | k tipi bir sunucuyu **kapatma maliyeti** (VARSAYILAN 0) | `ilp.yaml` |
| $L_k$ | **minimum çalışma süresi** (dilim sayısı; $L_k=1$ kısıtsız demektir) | `ilp.yaml` |
| $\alpha$ | güvenlik payı | `ilp.yaml` |
| $\rho^{cpu},\rho^{mem}$ | kullanılabilir kapasite oranı (VARSAYILAN 1,0) | `ilp.yaml` |
| $\hat D^{cpu}_t,\hat D^{mem}_t$ | tahmin edilen saatlik **tepe** talep | `forecast_*.parquet` |
| $x^0_k$ | ufuk başındaki çalışan sunucu sayısı | önceki günün planından |
| $\hat y_{k,\tau}$, $\tau \le 0$ | ufuktan önce (son $L_k-1$ dilimde) başlatılan sunucu sayısı | önceki günün planından |

**Karar değişkenleri (hepsi tamsayı):**
- $x_{k,t} \in \mathbb{Z}_{\ge 0}$: t diliminde çalışan k tipi sunucu sayısı
- $y_{k,t} \in \mathbb{Z}_{\ge 0}$: t diliminin başında **açılan** sunucu sayısı
- $z_{k,t} \in \mathbb{Z}_{\ge 0}$: t diliminin başında **kapatılan** sunucu sayısı

### 6.2 Model

**Amaç (toplam maliyet = çalışma + açma + kapatma):**

$$\min \; \sum_{t=1}^{H}\sum_{k\in K}\Big( c_k\,x_{k,t} \;+\; s_k\,y_{k,t} \;+\; d_k\,z_{k,t}\Big) \tag{1}$$

**Kapasite kısıtları (tahmin + güvenlik payı):**

$$\sum_{k\in K} \rho^{cpu} C_k\, x_{k,t} \;\ge\; (1+\alpha)\,\hat D^{cpu}_t \qquad \forall t \tag{2}$$

$$\sum_{k\in K} \rho^{mem} M_k\, x_{k,t} \;\ge\; (1+\alpha)\,\hat D^{mem}_t \qquad \forall t \tag{3}$$

**Durum geçişi (dönemler arası bağ):**

$$x_{k,t} - x_{k,t-1} \;=\; y_{k,t} - z_{k,t}, \qquad x_{k,0}=x^0_k \qquad \forall k,\; t=1,\dots,H \tag{4}$$

**Minimum çalışma süresi** (son $L_k$ dilimde açılan sunucuların hepsi t diliminde hâlâ çalışmak zorundadır):

$$\sum_{\tau=\max(1,\,t-L_k+1)}^{t} y_{k,\tau} \;+\; \sum_{\tau=t-L_k+1}^{0} \hat y_{k,\tau} \;\le\; x_{k,t} \qquad \forall k,\; t \tag{5}$$

(İkinci toplam yalnızca $t < L_k$ iken boş değildir ve önceki günden devralınan, süresi dolmamış başlatmaları temsil eder.)

**Tanım kümesi:**

$$x_{k,t},\,y_{k,t},\,z_{k,t} \in \mathbb{Z}_{\ge 0} \qquad \forall k,t \tag{6}$$

**İsteğe bağlı** (VARSAYILAN kapalı): filo üst sınırı $x_{k,t}\le N_k$ (kota).

### 6.3 Notlar

- Şablondaki statik model (denklem 3.1–3.4) bu modelin **özel durumudur** ($s_k=d_k=0,\; L_k=1$). Raporda Yöntem bölümünde genişletilmiş hâl verilecek. Yeni semboller ($y,z,s_k,d_k,L_k,\rho$) Sembol Listesi'ne eklenecek.
- Minimum **kapalı kalma** süresi **kapsam dışıdır** (bulutta yeni örnek istenildiği an açılabilir).
- $L_k=1$ iken (5), $z_{k,t}\le x_{k,t-1}$'e indirgenir (aynı dilimde açılıp kapanma yok).
- $s_k+d_k>0$ iken optimum çözümde aynı $(k,t)$ için $y$ ve $z$ birlikte pozitif olmaz. $s_k=d_k=0$ ise raporlanan açma sayısı $y_{k,t}=\max(0,\,x_{k,t}-x_{k,t-1})$ olarak yeniden hesaplanır.
- $s_k$ bir **modelleme parametresidir**, veri değildir. Önerilen biçim $s_k = c_k \cdot w$ ($w$: ısınma süresi saat eşdeğeri). **Duyarlılık ızgarası (VARSAYILAN): $w \in \{0,\,0.25,\,0.5,\,1\}$**, $L_k \in \{1,\,2,\,4\}$, $\alpha \in \{0,\,0.05,\,0.10,\,0.20\}$. Sonuçlar bu ızgarada **maliyet–yetersizlik eğrisi** olarak sunulur.
- **M7 ızgarası (kullanıcı kararı, 2026-11-01):** Tam faktöriyel (48 nokta), `ilp.yaml` `izgara`. Her noktada ILP (tüm modeller), kahin, R0, R1 ve R2'nin iki varyantı koşar. R2 eşikleri sabittir, uyumlu varyantın hedefi noktanın α'sına bağlıdır ($1/(1+\alpha)$; α = 0'da hedef = üst eşik = 1). Regret her noktanın kendi kahinine göredir. Kahin $(1+\alpha)D$'ye göre planlandığından ek olarak **alt sınır regret'i** (`regret_a0_yuzde`, aynı $w, L$'de α = 0 kahinine göre) raporlanır. Maliyet–yetersizlik eğrisi `pareto.csv` (günlük maliyet ↔ eksik saat oranı). Çıktılar `outputs/results/izgara/`.
- **Kayan pencere:** Her gün için model çözülür. Sonraki günün başlangıç durumu ($x^0_k,\hat y_{k,\tau}$) bir önceki günün **gerçekleşen** planından devralınır. Ufuk sonu etkisi (gün sonunda "hepsini kapat" eğilimi) bilinen bir sınırlılıktır, `horizon_h` / `commit_h` ayrı ayarlanabilir (VARSAYILAN 24/24).

### 6.4 Sunucu tipleri ve maliyet (`server_types.yaml`)
- Kaynak: **AWS EC2 on-demand, Linux** herkese açık fiyat listesi. **Fiyatlar bellekten yazılmayacak.** Betikle çekilip **çekildiği tarihle birlikte** dondurulur. Bölge: VARSAYILAN `eu-central-1` (kullanıcı değiştirebilir).
- Seçim kuralı: aynı nesil ve aynı vCPU boyutundan, **farklı GB/çekirdek oranlı** (hesaplama / genel amaçlı / bellek optimize) **3 tip.** Böylece CPU ve bellek kısıtları arasındaki denge (EDA'da incelenir) modelde anlam kazanır. Sürüm/tip adlarını fiyat listesinden doğrula.
- Maliyet **yalnızca sunucu saat ücretidir** (ağ, depolama, spot/rezerve indirimleri kapsam dışı).
- **M5 seçimi (VARSAYILAN, 2026-10-29):** `c8i.xlarge` / `m8i.xlarge` / `r8i.xlarge` (Intel Granite Rapids, 4 vCPU; 8 / 16 / 32 GiB → 2 / 4 / 8 GB/vCPU, Alibaba $r$ ızgarasıyla birebir). Fiyatlar `src/ilp/fiyat_cek.py` ile AWS Price List bulk API'sinden (yayın 2026-10-28) çekildi; aday satırların anlık görüntüsü `data/raw/aws/`. Talepteki "çekirdek" 1 vCPU'ya eşlenir.

### 6.5 Çözücü
- `pywraplp` + SCIP. Zaman sınırı `ilp.yaml`'da. Çözüm durumu `OPTIMAL` değilse hata ver, sessizce devam etme.
- Her çözümde kontrol: (2)–(5) sağlanıyor mu (bağımsız doğrulayıcı fonksiyon).
- **M5 ana koşu merkez değerleri (VARSAYILAN, `ilp.yaml`):** $\alpha=0{,}05$, $w=0{,}25$, $L_k=2$, $d_k=0$, $\rho=1$. Kayan pencere doğrulama günlerinde soğuk başlangıçla ($x^0=0$) ısınır, ölçütler yalnız test günlerinde hesaplanır.

---

## 7. Karşılaştırma Stratejileri ve Başarım Ölçütleri

Tüm stratejiler **aynı gerçek talep** `D_t` üzerinde değerlendirilir.

| Strateji | Tanım |
|---|---|
| **Tahmine dayalı ILP** (ana yöntem) | Bölüm 6, girdi: model tahmini $\hat D_t$ (her tahmin modeli için ayrı). |
| **Kahin (oracle) ILP** | Aynı ILP, girdi: **gerçek** $D_t$ (mükemmel öngörü). Alt sınır, regret hesabı için. |
| **Tepkisel-ILP (R0)** | Her dilimde ufuk=1 ile çözülür, girdi: **bir önceki dilimin gerçekleşen tepesi** $D_{t-1}$. Tahminin değerini yalıtır. |
| **HPA benzeri eşik kuralı (R1)** | Kullanım $u_{t}=\max(D^{cpu}_{t-1}/\text{kap}^{cpu}_{t-1},\,D^{mem}_{t-1}/\text{kap}^{mem}_{t-1})$. $u>\tau_{üst}$ ise tek bir **varsayılan ölçek tipinde** sunucu ekle (hedef $\tau_{hedef}$'e dek). $u<\tau_{alt}$ ve soğuma süresi dolduysa sunucu çıkar ($L_k$'ye uyarak). Eşikler `ilp.yaml`'da (VARSAYILAN: üst 0,8 / hedef 0,7 / alt 0,5 / soğuma 2 dilim). |
| **Hibrit: tahmine dayalı ILP + tepkisel tampon (R2)** | Taban: M5'in tahmine dayalı ILP planı (her tahmin modeli için ayrı). Üzerine R1 ile aynı kural ve eşiklerle yönetilen, ölçek tipinde bir **tampon** eklenir. Tampon büyüklüğü, $t$'deki plan kapasitesiyle birlikte $D_{t-1}$'i hedef kullanımda karşılayan en küçük sayıdır. Tampon 0'ın altına inmez (sunucu sayısı plandan az olmaz), açılışları $L_k$'ye uyar. Maliyet: plan + tampon (açılışlar ayrı sayılır). |
| **Hibrit, plana uyumlu eşikler (R2-uyumlu)** | R2 ile aynı mekanizma, eşikler `ilp.yaml` `tepkisel.hibrit_uyumlu` (VARSAYILAN: üst 1,0 / hedef $1/(1+\alpha)$ / alt 0,8 / soğuma 2). Kullanıcı kararı (2026-11-01, "c"): R2 iki varyantla ayrı satırlar olarak raporlanır. Gerekçe: ILP planları ≈ $1/(1+\alpha)$ kullanım hedeflediğinden HPA eşiği (0,8) neredeyse her dilimde tetiklenir ve tampon alt eşiğe inmediği için küçülmez. |

**M6 kullanıcı kararları (2026-10-31):** Kullanıcı M6 talebinde eşik kuralını "R1", hibriti "R2" olarak adlandırdı. Önceki "Tepkisel-ILP (R1)" tanımı **R0** adıyla korunur, önceki "eşik kuralı (R2)" artık **R1**'dir. Hibrit yeni eklendi. Varsayılan ölçek tipi `m8i.xlarge` (VARSAYILAN, `ilp.yaml` `tepkisel.olcek_tipi`); R1'in alt sınırı 1 sunucu. Tüm tepkisel stratejiler ILP ile aynı $s_k, d_k, L_k$ maliyet parametrelerini kullanır. Simülasyon doğrulama döneminin ilk diliminden soğuk başlangıçla ($x=0$) ısınır, ölçütler yalnız test dilimlerinde hesaplanır. Çıktılar `outputs/reactive/` altına yazılır (kullanıcı talebi).

**Ölçütler (hepsi gerçek `D_t` üzerinden, α payı **dahil edilmeden**):**
1. **Toplam maliyet** (çalışma + açma + kapatma)
2. **Yetersiz kalan dilim sayısı** ve **karşılanmayan talep** (çekirdek-saat, GB-saat)
3. **Boşta kapasite oranı:** $\overline{(\text{kapasite}-D)/\text{kapasite}}$ (uygulamada pay $\max(\text{kapasite}-D,0)$: yetersiz dilimde boşta kapasite 0 sayılır, açık ayrıca ölçüt 2'de)
4. **Açma sayısı** (toplam başlatılan sunucu)
5. **Regret:** strateji maliyeti − kahin maliyeti
5a. **Tepki gecikmesi (kullanıcı kararı, 2026-10-31, M6):** art arda eksik kapasiteli dilimler bir **olay** sayılır. Olay sayısı, ortalama olay süresi ve en uzun olay süresi (dilim) raporlanır. Bu ölçüt, açığın kaç saatte kapatıldığını gösterir. Ek açılış gecikmesi modellenmez (0); kararların $D_{t-1}$'e dayanması zaten 1 dilimlik gecikme yaratır.
6. Ayrıca her tahmin modeli için **tahmin hatası ↔ karar maliyeti** ilişkisi (aynı tabloda).

Sonuçlar havuzlar üzerinden ortalama ± standart sapma. $M_{ref}$ ve ($\alpha$, $w$, $L_k$) ızgaraları için ayrı tablolar.

---

## 8. Sentetik Senaryolar (`outputs/synthetic/`, gerçek sonuçlardan ayrı)

Gerçek veride bulunmayan durumlar kontrollü üretilir (tohumlu): **ani yük artışı** (basamak ve darbe), **ani düşüş**, **mevsimsellik bozulması**. Aynı boru hattı çalışır, tablolar "sentetik" olarak ayrı etiketlenir.

**M7 kullanıcı kararları (2026-11-01):** Bozulma **yalnız test dönemine**, 5 dk havuz toplamı serisine uygulanır ve saatlik tepe yeniden hesaplanır. Modeller bozulmamış eğitim+doğrulama verisiyle (M4'te seçilen SARIMA derecesi, LSTM en iyi epoch sayısı) yeniden eğitilir. Bozulmuş seride parametreler sabit tutularak walk-forward tahmin yapılır. Ardından ILP, R0, R1 ve R2 merkez parametrelerle koşar. Tüm havuzlar kullanılır. Senaryolar (VARSAYILAN, `configs/synthetic.yaml`): basamak ×1,5 (rastgele başlangıçtan test sonuna dek), darbe ×2 (3 saat), ani düşüş ×0,5 (test sonuna dek), mevsimsellik bozulması (test dilimlerinde değer 12 saat öncekiyle değiştirilir). Ek olarak bozulmasız bir **kontrol** senaryosu, yeniden üretimin M4/M6 ile tutarlılığını ölçer. Başlangıç saati her (senaryo, veri seti) için tohumla seçilir ve aynı veri setinin tüm havuzlarında ortaktır.

---

## 9. Kilometre Taşları ve Bitti Kriterleri

| Taş | İçerik | Bitti kriteri |
|---|---|---|
| **M0** | Ortam, smoke testler | CUDA, SCIP, SARIMAX, Parquet okuma/yazma çalışıyor |
| **M1** | Bitbrains ingest + ön işleme | `pool_*.parquet` üretildi, `quality_report.json` var, zaman birimi/sütun doğrulamaları geçti |
| **M2** | EDA (Bölüm 10) | Şekiller ve tablolar `outputs/eda/` altında |
| **M3** | Alibaba ingest + ön işleme ($M_{ref}$ ızgarası) | Aynı çıktılar, 3 r değeri için |
| **M4** | P0–P2 tahmin modelleri | Walk-forward tahmin dosyaları + ölçüt tablosu |
| **M5** | ILP modeli + testler | Birim testleri geçti (Bölüm 11), çözümler doğrulayıcıdan geçti |
| **M6** | Tepkisel yöntemler + değerlendirme | Bölüm 7 tabloları üretildi |
| **M7** | Duyarlılık ve sentetik senaryolar | Izgaralar ve `outputs/synthetic/` tamam |
| **M8** | Raporlama çıktıları | Şekil/tablo dosyaları Şekil 4.x / Tablo 4.x numaralandırmasına hazır |

---

## 10. EDA İçeriği (M2)

| Analiz | Karara bağlanan şey |
|---|---|
| Kapsama, boşluk, VM yaşam süresi | Enterpolasyon kuralı, havuz oluşturma |
| Toplam talep zaman grafikleri, **saatlik tepe/ortalama oranı** | Tepe tabanlı planlamanın gerekçesi |
| ACF/PACF, periodogram, STL ayrışımı | SARIMA mevsim uzunluğu |
| ADF ve KPSS testleri | ARIMA `d` |
| Talep CPU/bellek oranı ↔ sunucu tiplerinin GB/çekirdek oranı | Hangi kaynağın kısıtı baskın |
| Mevsimsel naif hata | Tahmin edilebilirlik tabanı |
| Bitbrains'te VM bazında ve toplamda günlük otokorelasyon | Toplamda mevsimselliğin ne kadar kaldığı |
| (Opsiyonel) İstenen vs kullanılan kaynak (Alibaba `container_meta`, Bitbrains provisioned vs usage) | "İsraf" motivasyon grafiği |

---

## 11. Testler (`tests/`)

- **Ön işleme:** Toplamın tepesi ≠ tepelerin toplamı (küçük elle hesaplanmış örnek); birim dönüşümü; enterpolasyon limiti; yaşam süresi dışı = 0.
- **Sızıntı:** Ölçekleyici istatistikleri yalnızca eğitim verisinden; walk-forward'da gelecek veri kullanılmıyor.
- **ILP:** (a) Basit örnekte bilinen optimum. (b) **Minimum çalışma süresi:** talep kısa süreli sıçrayıp düştüğünde sunucular en az $L_k$ dilim açık kalıyor. (c) Başlatma maliyeti yüksekken sunucu sayısı dalgalanmıyor. (d) $s_k=d_k=0, L_k=1$ iken dönem-dönem bağımsız çözümle aynı maliyet. (e) Doğrulayıcı (2)–(5)'i bağımsız denetliyor. (f) Küçük örnekte SCIP ile CP-SAT aynı optimum. (g) **LP gevşetmesi + yukarı yuvarlama** ile ILP maliyet farkı raporlanıyor (rapordaki "yuvarlama her zaman en iyi çözümü vermez" iddiasının kanıtı).
- **Değerlendirme:** Yetersizlik sayacı gerçek `D` ve α'sız kapasite ile çalışıyor.

---

## 12. Kapsam Dışı (bilinçli sınırlılıklar, raporda açıkça yazılacak)

- Pod düzeyi yerleştirme (bin packing) ve parçalanma; veriler VM/makine düzeyinde olup bir Kubernetes **node havuzunun toplam kaynak talebi için vekil (proxy)** olarak kullanılır.
- CPU ve bellek dışındaki kaynaklar (ağ, disk, GPU).
- Spot/rezerve fiyatlandırma, ağ çıkışı, depolama maliyeti.
- Minimum kapalı kalma süresi, stokastik programlama (ileriye yönelik öneri olarak anılacak).
- Alibaba'da fiziksel bellek (M_ref varsayımı), 8 günlük veri nedeniyle haftalık mevsimsellik; ilk 2 günün veri kalitesi nedeniyle atılması (kullanılan 6 gün, test yalnızca 1 gün).

---

## 13. Rapor Bağlantısı (bilgi)

**M8 (2026-11-01):** `src/rapor.py`, `configs/rapor.yaml` katalog sırasına göre bölüm başına numaralanmış PNG/CSV üretir. Alt başlıklar şablonun 4. bölümünden alınır (Tahmin Sonuçları, Kapasite Planlama Sonuçları, Otomatik Ölçeklendirme ile Karşılaştırma, Kontrollü Sentetik Senaryolar). Şablondaki Tablo 3.1 (örnek değerli sunucu tipleri) gerçek AWS değerleriyle üretilir. Tablolarda ondalık virgül kullanılır. Ana karşılaştırma tablolarında R2 yalnız SARIMA (P1) tabanıyla verilir, çünkü şablon tahmin yöntemi olarak ARIMA/SARIMA'yı adlandırıyor; diğer modeller `outputs/reactive/` tablolarındadır.

Çıktılar, DD-050 formatına ve `Tasarim_Uygulamasi_Rapor_Sablonu.docx` şablonuna **en son adımda toplu olarak** entegre edilecek. Ajan Word dosyası üretmez. Şekil/tablo dosyalarını (PNG/CSV) ve bu dosyadaki sembol/denklem numaralandırmasını (1)–(6) korur. Rapordaki Yöntem bölümü denklemleri (3.x) bu denklemlerden türetilecektir.