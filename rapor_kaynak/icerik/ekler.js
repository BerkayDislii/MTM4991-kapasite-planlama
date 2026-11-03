// EKLER: A (kod yapısı), B (Alibaba SARIMA seçimleri), C (duyarlılık ızgarası özeti)
const { csvOku, tr } = require("./veri.js");

module.exports = (h) => {
  const { P, H1, H2, tabloG, tabloAlt, bosluk } = h;
  const KUCUK = 18;

  const kod = [
    ["src/ingest.py, src/ingest_alibaba.py", "Ham verinin okunması, sütun ve zaman birimi doğrulaması, 5 dk ızgaraya hizalama"],
    ["src/preprocess.py, src/preprocess_alibaba.py", "Birim dönüşümü, yaşam süresi ve eksik veri kuralları, havuz toplamı ve saatlik tepe; M_{ref} ızgarası"],
    ["src/eda.py", "Keşifsel analiz: tepe/ortalama oranı, ACF/PACF, STL, ADF/KPSS, kaynak oranı"],
    ["src/forecast/naif.py", "P0 taban modelleri (mevsimsel naif, son değer)"],
    ["src/forecast/sarima.py", "P1 SARIMA: derece seçimi, eğitim+doğrulama ile son uydurma, ileri yürüyen tahmin"],
    ["src/forecast/lstm.py", "P2 LSTM: iki aşamalı eğitim, erken durdurma, üç tohum"],
    ["src/forecast/ortak.py, veri.py, ozet.py", "Havuz tanımları, tahmin ölçütleri, özet tablolar"],
    ["src/ilp/fiyat_cek.py", "AWS fiyat listesinden sunucu tiplerinin çekilmesi ve tarihli olarak saklanması"],
    ["src/ilp/model.py", "Çok dönemli ILP (SCIP), bağımsız doğrulayıcı, CP-SAT çapraz kontrolü, LP gevşetmesi + yuvarlama"],
    ["src/ilp/solve.py", "Kayan pencere ile günlük çözüm, kahin planı, LP karşılaştırması"],
    ["src/evaluate.py", "Gerçek talep üzerinde maliyet, eksik kapasite, boşta kapasite ve olay ölçütleri"],
    ["src/baselines/tepkisel.py", "R0, R1 ve R2 stratejilerinin benzetimi ve regret hesabı"],
    ["src/izgara.py", "(α, w, L) duyarlılık ızgarası (48 nokta)"],
    ["src/sentetik.py", "Sentetik senaryoların üretimi ve değerlendirilmesi"],
    ["src/rapor.py", "Rapor şekil ve tablolarının üretimi"],
    ["configs/*.yaml", "Tüm sayısal parametreler (ön işleme, tahmin, ILP, eşikler, senaryolar, rapor)"],
    ["tests/", "Birim testleri: ön işleme, tahmin (sızıntı denetimi), ILP (a)–(g), tepkisel stratejiler, sentetik senaryolar, rapor"],
  ];

  // Ek B: Alibaba SARIMA (dereceler r'den bağımsız; r = 4 satırları)
  const ab = csvOku("alibaba_sarima_secim.csv");
  const tB = ab.map((r) => [r.havuz.replace("alibaba_", "").replace("_r4", ""), r.kaynak === "cpu" ? "İşlemci" : "Bellek", r.d, r.D,
    tr(r.stl_mevsimsellik_gucu, 3), r.secilen_aile === "mevsimsel" ? "Mevsimsel" : "Mevsimsel olmayan",
    `SARIMA${r.order} × ${r.seasonal_order}`]);

  // Ek C: ızgara özeti
  const grupAd = { bitbrains_fastStorage: "Bitbrains fastStorage", bitbrains_rnd: "Bitbrains Rnd", bitbrains_fastStorage_alt: "fastStorage alt havuzları",
    alibaba_r2: "Alibaba (r = 2)", alibaba_r4: "Alibaba (r = 4)", alibaba_r8: "Alibaba (r = 8)" };
  const strAd = { ilp: "Tahmine dayalı ILP (SARIMA)", r0_tepkisel_ilp: "R0 Tepkisel-ILP", r1_esik: "R1 Eşik kuralı", r2_hibrit: "R2 Hibrit (HPA)", r2_hibrit_uyumlu: "R2 Hibrit (uyumlu)" };
  const ic = csvOku("izgara_ozet.csv");
  const tC = ic.map((r) => [grupAd[r.grup], strAd[r.strateji], `${tr(r.eksik_min, 1)} – ${tr(r.eksik_max, 1)}`, `${tr(r.regret_min, 1)} – ${tr(r.regret_max, 1)}`]);

  return [
    H1("EKLER"),
    P("Bu bölümde, tez metninde yer alması gerekmeyen ancak çalışmanın yeniden üretilebilmesi ve sonuçların ayrıntılı incelenmesi için yararlı olan bilgiler verilmiştir."),

    H2("Ek A Proje Kod Yapısı ve Yeniden Üretim Adımları"),
    P("Çalışmanın tüm adımları Python 3 ile yazılmış bir kod tabanında gerçekleştirilmiştir. Kullanılan başlıca kütüphaneler pandas, polars, NumPy, statsmodels, PyTorch, Google OR-Tools ve matplotlib'tir. Kod tabanının modülleri ve görevleri Tablo A.1'de verilmiştir. Tüm sayısal parametreler kodun içine gömülmemiş, configs klasöründeki YAML dosyalarında tutulmuştur."),
    tabloAlt("A.1", "Proje modülleri ve görevleri"),
    tabloG([3300, 5200], ["Modül", "Görev"], kod, { size: KUCUK }),
    bosluk(),
    P("Sonuçlar, modüllerin python -m src.<modül> biçiminde şu sırayla çalıştırılmasıyla yeniden üretilebilir: ingest, preprocess ve eda (Bitbrains); ingest_alibaba ve preprocess_alibaba (Alibaba); forecast.naif, forecast.sarima ve forecast.lstm (tahmin); ilp.fiyat_cek ve ilp.solve (kapasite planlama); baselines.tepkisel (karşılaştırma stratejileri); izgara ve sentetik (duyarlılık ve sentetik senaryolar); rapor (şekil ve tablolar). Birim testleri pytest ile çalıştırılır. Ham veri dosyaları proje deposuna dahil edilmemiştir; Bitbrains ve Alibaba veri kümeleri yayımlandıkları kaynaklardan ayrıca indirilmelidir."),

    H2("Ek B Alibaba Havuzlarında Seçilen SARIMA Modelleri"),
    P("Alibaba havuzlarında seçilen SARIMA modelleri Tablo B.1'de verilmiştir. r katsayısı bellek serisini yalnızca sabit bir çarpanla ölçeklediğinden, seçilen dereceler üç r değeri için de aynıdır; tabloda r = 4 sonuçları gösterilmiştir. İşlemci serilerinin tamamında mevsimsellik gücü 0,64 eşiğinin üzerinde olduğundan D = 1 alınmıştır. p3 havuzunun işlemci serisinde ise doğrulama hatası daha düşük olan mevsimsel olmayan aile seçilmiştir."),
    tabloAlt("B.1", "Alibaba havuzlarında seçilen SARIMA modelleri"),
    tabloG([900, 1000, 500, 500, 1000, 1800, 2800], ["Havuz", "Kaynak", "d", "D", "STL F_{s}", "Seçilen aile", "Model"], tB, { size: KUCUK, birlestir: 1 }),
    bosluk(),

    H2("Ek C Duyarlılık Izgarasının Özet Sonuçları"),
    P("Duyarlılık ızgarasının (α, w, L; 48 nokta) özet sonuçları Tablo C.1'de verilmiştir. Tabloda her grup ve strateji için 48 ızgara noktasında elde edilen eksik kapasiteli saat oranı ile α = 0 kahinine göre regret değerlerinin en küçük ve en büyük değerleri gösterilmiştir. Tahmine dayalı ILP ve hibrit stratejiler için SARIMA tahmini kullanılmıştır. R1 eşik kuralı α'dan bağımsız olduğundan, bu stratejideki değişim yalnızca açma maliyeti ve minimum çalışma süresinden kaynaklanmaktadır. Uyumlu eşikli R2, tüm gruplarda ILP'ye göre daha dar ve daha düşük bir eksik kapasite aralığında kalmıştır. Tüm ızgara sonuçları proje deposundaki outputs/results/izgara klasöründedir."),
    tabloAlt("C.1", "Duyarlılık ızgarasında stratejilerin eksik kapasite ve regret aralıkları"),
    tabloG([2100, 2400, 2000, 2000], ["Grup", "Strateji", "Eksik saat aralığı (%)", "Regret aralığı, α = 0 kahinine göre (%)"], tC, { size: KUCUK, birlestir: 1 }),
    bosluk(),
  ];
};
