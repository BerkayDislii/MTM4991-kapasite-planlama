# Canlı Kapasite Planlayıcı (Kubernetes demosu)

Raporun kapasite planlama sistemini gerçek bir Kubernetes kontrol düzleminde çalıştırır. Düğümler
[KWOK](https://kwok.sigs.k8s.io/) ile oluşturulan sahte makinelerdir, pod'ları gerçek `kube-scheduler`
yerleştirir. Eksik kapasite, raporda bir yüzde olarak görünür; burada ise yer bulamayan (Pending) pod olarak.

```
talep izi (5 dk) ──► yük oynatıcı ──► pod sayısı ──► Kubernetes ──► yer bulamayan pod = eksik kapasite
tahmin (gün başı) ─► ILP (src/ilp) ─► düğüm planı ─┐
gözlenen tepe (saatlik) ─► R2 tamponu ─────────────┴─► düğüm sayıları (c8i / m8i / r8i)
```

## Kurulum (bir kez)

```bash
cd ~/Masaüstü/MTM4991_Proje
.venv/bin/pip install -r demo/requirements.txt     # kubernetes istemcisi + streamlit
bash demo/k8s/kume_kur.sh                          # kind kümesi + KWOK + hız ayarı (tekrar çalıştırılabilir)
```

## Çalıştırma

İki terminal açın (ikisi de proje kökünde):

```bash
# Terminal 1: demo (varsayılan: fastStorage, SARIMA, ILP + R2 tamponu, test günü 24, 1 gün ≈ 6 dk)
.venv/bin/python -m demo.calistir

# Terminal 2: canlı panel → http://localhost:8501
.venv/bin/streamlit run demo/panel.py --client.toolbarMode minimal --browser.gatherUsageStats false
```

Sık kullanılan seçenekler (`.venv/bin/python -m demo.calistir --yardim`):

| Seçenek | Anlamı |
|---|---|
| `--strateji ilp` | Tamponu kapat: yalnız tahmine dayalı ILP planı |
| `--gun-sayisi 6 --hizli` | Tüm test dönemi, beklemeden (doğrulama için) |
| `--saat-saniye 6` | Daha hızlı oynat (1 benzetim saati = 6 s) |
| `--kuru` | Kubernetes olmadan, bellek içi kümeyle (hesap denetimi) |
| `--kayitli-plan` | ILP'yi canlı çözmek yerine raporun kayıtlı planını kullan |

## Doğrulama

```bash
.venv/bin/python -m demo.dogrula               # son koşuyu raporun benzetimiyle karşılaştırır
.venv/bin/python -m pytest demo/test_demo.py -q
```

`dogrula` her saat için düğüm sayılarını, eksik kapasiteli saatleri ve maliyeti benzetimle karşılaştırır.
Tüm test dönemi (`--gun-sayisi 6`) için raporun Tablo 4.9 değerleri yeniden elde edilir: R2 (uyumlu)
927 USD/gün ve %8,3 eksik saat; yalnız ILP 887 USD/gün ve %15,3 eksik saat.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `ayar.yaml` | Havuz, model, strateji, gün, birim pod, hız |
| `cekirdek.py` | Veri, ILP planlayıcı (gün gün) ve R2 tamponu (saat saat); Kubernetes'ten bağımsız |
| `kume.py` | Küme katmanı: `KubeKume` (gerçek API) ve `SahteKume` (bellek içi) |
| `calistir.py` | Ana döngü ve CSV kayıtları (`cikti/adim.csv`, `cikti/saat.csv`, `cikti/durum.json`) |
| `dogrula.py` | Demoyu benzetimle karşılaştırır |
| `panel.py` | Streamlit paneli |
| `k8s/kume_kur.sh` | Küme kurulumu |
| `k8s/ornek-*.yaml` | Elle deneme için tek düğüm ve 5 pod'luk örnek |

## Birim pod neden 2 çekirdek / 4 GiB?

Talep, sabit boyutlu pod'lara bölünür: işlemci pod'u yalnız 2 çekirdek, bellek pod'u yalnız 4 GiB ister.
Bu boyutlar her sunucu tipinin kapasitesine (4 çekirdek; 8/16/32 GiB) tam bölünür. Bu yüzden pod
yerleştirmesinde boşa giden parça kalmaz ve "yer bulamayan pod var" durumu ancak ve ancak "talep >
kapasite" olduğunda ortaya çıkar. Bu, raporun eksik kapasite tanımıyla birebir aynıdır
(`test_birim_pod_esdegerligi`).
