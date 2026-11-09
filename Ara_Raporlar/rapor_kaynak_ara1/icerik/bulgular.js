// 4. BULGULAR VE TASARIM ÇIKTILARI — metin içeriği. Tablolar outputs/rapor ve outputs/eda CSV'lerinden üretilir.
const { csvOku, tr, ortStd } = require("./veri.js");

module.exports = (h) => {
  const { P, H1, H2, tabloG, tabloAlt, sekil, bosluk } = h;
  const madde = (t, son = false) => P(t, { para: { indent: { left: 567 }, spacing: { after: son ? 360 : 120 } } });
  const not = (t) => P(t, { run: { size: 20 }, para: { spacing: { before: 0, after: 360 } } });

  // ---------- Tablo 4.1: toplam tepe / toplam ortalama ----------
  const to = csvOku("tablo_tepe_ortalama.csv");
  const deger = (havuz, donem, k) => Number(to.find((r) => r.havuz === havuz && r.donem === donem && r.kaynak === k).toplam_tepe_toplam_ortalama);
  const donemler = ["egitim", "dogrulama", "test"], kaynaklar = ["cpu", "mem"];
  const altlar = [0, 1, 2, 3, 4].map((i) => `bitbrains_fastStorage_s${i}`);
  const t41 = [
    ["Bitbrains fastStorage", ...donemler.flatMap((dn) => kaynaklar.map((k) => tr(deger("bitbrains_fastStorage", dn, k), 3)))],
    ["Bitbrains Rnd", ...donemler.flatMap((dn) => kaynaklar.map((k) => tr(deger("bitbrains_rnd", dn, k), 3)))],
    ["fastStorage alt havuzları", ...donemler.flatMap((dn) => kaynaklar.map((k) => ortStd(altlar.map((a) => deger(a, dn, k)), 3)))],
    ["Alibaba (8 havuz)", "1,171", "1,022", "—", "—", "—", "—"],
  ];

  // ---------- Tablo 4.2: talep bileşimi ve bağlayıcı kaynak ----------
  const ko = csvOku("tablo_kaynak_orani.csv");
  const kr = (havuz) => ko.find((r) => r.havuz === havuz);
  const yuz = (x) => tr(100 * Number(x), 1);
  const t42 = [
    ...[["bitbrains_fastStorage", "Bitbrains fastStorage"], ["bitbrains_rnd", "Bitbrains Rnd"]].map(([hv, ad]) => {
      const r = kr(hv);
      return [ad, tr(r.oran_medyan, 2), tr(r.oran_p5, 2), tr(r.oran_p95, 2), yuz(r.bellek_baglayici_orani_r2), yuz(r.bellek_baglayici_orani_r4), yuz(r.bellek_baglayici_orani_r8)];
    }),
    ["fastStorage alt havuzları", ortStd(altlar.map((a) => Number(kr(a).oran_medyan))), ortStd(altlar.map((a) => Number(kr(a).oran_p5))),
      ortStd(altlar.map((a) => Number(kr(a).oran_p95))), ortStd(altlar.map((a) => 100 * Number(kr(a).bellek_baglayici_orani_r2)), 1),
      ortStd(altlar.map((a) => 100 * Number(kr(a).bellek_baglayici_orani_r4)), 1), ortStd(altlar.map((a) => 100 * Number(kr(a).bellek_baglayici_orani_r8)), 1)],
    // quality_report.json → mref_alibaba (eğitim dönemi, 8 havuz ortalaması)
    ["Alibaba (r = 2)", "4,38", "3,03", "6,20", "100,0", "61,5", "0,0"],
    ["Alibaba (r = 4)", "8,75", "6,06", "12,39", "100,0", "100,0", "61,5"],
    ["Alibaba (r = 8)", "17,50", "12,11", "24,79", "100,0", "100,0", "100,0"],
  ];

  // ---------- Tablo 4.3: tahmin ölçütleri (Alibaba yalnız r = 4) ----------
  const tm = csvOku("tablo_4_1_tahmin_olcutleri.csv").filter((r) => !/Alibaba \(r = [28]\)/.test(r.Grup));
  const grupAd = (g) => g.replace("Bitbrains fastStorage alt havuzları", "fastStorage alt havuzları");
  const t43 = tm.map((r) => [grupAd(r.Grup), r.Kaynak, r.Model, r.MAE, r.RMSE, r["MAPE (%)"], r.MASE, r["Eksik tahmin (%)"]]);

  // ---------- Tablo 4.4: SARIMA seçimi ----------
  const ss = csvOku("tablo_4_2_sarima_secim.csv");
  const t44 = ss.map((r) => [r.Havuz.replace("_", " "), r.Kaynak, r.d, r.D, r["STL F_s"], r["Seçilen aile"], r.Model.replace("×", " × ")]);

  // ---------- Tablo 4.5: ILP sonuçları (Alibaba yalnız r = 4) ----------
  const il = csvOku("tablo_4_3_ilp_sonuclari.csv").filter((r) => !/Alibaba \(r = [28]\)/.test(r.Grup));
  const strAd = (r) => (r.Strateji === "Kahin ILP" ? "Kahin ILP" : r["Tahmin modeli"]);
  const t45 = il.map((r) => [grupAd(r.Grup), strAd(r), r["Günlük maliyet (USD)"], r["Regret (%)"], r["Eksik saat (%)"],
    r["Karşılanmayan işlemci (%)"], r["Karşılanmayan bellek (%)"], r["En uzun olay (sa)"], r["Açılış"]]);

  // ---------- Tablo 4.6: LP yuvarlama ----------
  const lp = csvOku("tablo_4_4_lp_yuvarlama.csv");
  const t46 = lp.map((r) => [r["Veri seti"], r["Çözüm sayısı"], r["Yuvarlanmış plan (5)'e uygun (%)"], r["Yuvarlama maliyet farkı, ort. (%)"],
    r["Yuvarlama maliyet farkı, en büyük (%)"], r["LP alt sınır boşluğu, ort. (%)"], r["ILP çözüm süresi, ort. (s)"], r["ILP çözüm süresi, en büyük (s)"]]);

  // ---------- Tablo 4.7: α duyarlılığı ----------
  const al = csvOku("tablo_4_5_alfa_duyarlilik.csv");
  const t47 = al.map((r) => [grupAd(r.Grup), r["α"], r["Günlük maliyet (USD)"], r["Regret, aynı α kahinine göre (%)"],
    r["Regret, α = 0 kahinine göre (%)"], r["Eksik saat (%)"], r["En uzun olay (sa)"]]);

  // ---------- Tablo 4.8: w ve L etkisi ----------
  const wl = csvOku("tablo_4_6_w_L_etkisi.csv");
  const t48 = wl.map((r) => [grupAd(r.Grup), r.w, r["Açılış (L = 1)"], r["Açılış (L = 2)"], r["Açılış (L = 4)"], r["Günlük maliyet, L = 2 (USD)"], r["Eksik saat, L = 2 (%)"]]);

  // ---------- Tablo 4.9 / 4.10: strateji karşılaştırması ----------
  const secili = (r) => r.Strateji !== "Tahmine dayalı ILP" || r["Tahmin modeli"] === "SARIMA (P1)";
  const sAd = (r) => (r.Strateji === "Tahmine dayalı ILP" ? "Tahmine dayalı ILP (SARIMA)" : r.Strateji.replace(" (HPA eşikleri)", " (HPA)").replace(" (uyumlu eşikler)", " (uyumlu)"));
  const kars = (dosya) => csvOku(dosya).filter(secili).map((r) => [grupAd(r.Grup), sAd(r), r["Günlük maliyet (USD)"], r["Regret (%)"],
    r["Eksik saat (%)"], r["Karşılanmayan işlemci (%)"], r["Karşılanmayan bellek (%)"], r["En uzun olay (sa)"], r["Açılış"]]);
  const t49 = kars("tablo_4_7_karsilastirma_bitbrains.csv");
  const t410 = kars("tablo_4_8_karsilastirma_alibaba.csv");

  // ---------- Tablo 4.11: sentetik özet (senaryo × strateji) ----------
  const so = csvOku("tablo_4_9_sentetik_ozet.csv");
  const stratejiler = ["Tahmine dayalı ILP", "R0 Tepkisel-ILP", "R1 Eşik kuralı", "R2 Hibrit (HPA eşikleri)", "R2 Hibrit (uyumlu eşikler)"];
  const senaryolar = [...new Set(so.map((r) => r.Senaryo))];
  const sv = (vs, sen) => so.find((r) => r["Veri seti"] === vs && r.Senaryo === sen);
  const t411 = senaryolar.flatMap((sen) => stratejiler.map((st) => [sen, st.replace(" (HPA eşikleri)", " (HPA)").replace(" (uyumlu eşikler)", " (uyumlu)"),
    sv("Bitbrains", sen)[`${st}: regret (%)`], sv("Bitbrains", sen)[`${st}: eksik saat (%)`],
    sv("Alibaba", sen)[`${st}: regret (%)`], sv("Alibaba", sen)[`${st}: eksik saat (%)`]]));

  // ---------- Tablo 4.12: sentetik MASE (senaryo × model) ----------
  const sm = csvOku("tablo_4_10_sentetik_tahmin.csv");
  const modeller = ["SARIMA (P1)", "LSTM (P2)", "Mevsimsel naif (P0)", "Son değer (P0)"];
  const smv = (vs, sen) => sm.find((r) => r["Veri seti"] === vs && r.Senaryo === sen);
  const t412 = senaryolar.flatMap((sen) => modeller.map((md) => [sen, md,
    smv("Bitbrains", sen)[`İşlemci: ${md}`], smv("Bitbrains", sen)[`Bellek: ${md}`],
    smv("Alibaba", sen)[`İşlemci: ${md}`], smv("Alibaba", sen)[`Bellek: ${md}`]]));

  const KUCUK = 18; // 9 punto (büyük tablolar)
  return [
    H1("4. BULGULAR VE TASARIM ÇIKTILARI"),
    P("Bu bölümde Bölüm 3'te tanımlanan deneylerin sonuçları sunulmuş ve tartışılmıştır. Önce veri kümelerinin keşifsel analizi verilmiş, ardından tahmin modellerinin başarımı, ILP planlarının maliyet ve kapasite yeterliliği, tepkisel ve hibrit stratejilerle karşılaştırma ve sentetik senaryo sonuçları incelenmiştir. Aksi belirtilmedikçe tüm sonuçlar test dönemine aittir ve merkez parametrelerle (α = 0,05, w = 0,25, L = 2) elde edilmiştir. Birden çok havuzu olan gruplarda değerler havuzlar üzerinden ortalama ± standart sapma, LSTM için ise havuz ve tohumlar üzerinden ortalama ± standart sapma olarak verilmiştir. Tablolarda ondalık ayırıcı olarak virgül kullanılmıştır."),

    // ===== 4.1
    H2("4.1 Veri Kümelerinin Keşifsel Analizi"),
    P("Bitbrains fastStorage havuzunun saatlik tepe ve saatlik ortalama talebi Şekil 4.1'de görüldüğü gibidir. Talep, gün içi döngülerden çok birkaç gün süren düzey değişimleriyle karakterize edilir. İşlemci talebi eğitim döneminde yaklaşık 100 ile 1.200 çekirdek arasında değişmiş, test döneminde ise daha düşük bir düzeye inmiştir. Bellek talebinde kısa süreli ve yüksek sıçramalar dikkat çekmektedir. Test döneminde bu sıçramaların genliği eğitim dönemindekinden büyüktür; örneğin bellek tepe talebi test döneminde 2.500 GiB'ı aşmıştır. Test döneminin eğitim döneminden farklı bir davranış göstermesi, tahmin modellerini zorlayan önemli bir etkendir."),
    ...sekil("sekiller/talep_zaman_bitbrains_fastStorage.png", 1924, 1564, 15, "4.1", "Bitbrains fastStorage havuzunun saatlik tepe ve saatlik ortalama işlemci ve bellek talebi"),
    P("Tepe talebin ortalama talebe oranı Tablo 4.1'de verilmiştir. Oran, her dönemde saatlik tepe değerlerin toplamının saatlik ortalama değerlerin toplamına bölünmesiyle hesaplanmıştır. Tüm havuz ve dönemlerde oran 1'den büyüktür; saat içi tepe, saatlik ortalamayı Bitbrains'te %7 ile %37, Alibaba'da işlemci için %17 ve bellek için %2 oranında aşmıştır. Oran, fastStorage havuzunun test döneminde en yüksek değerine ulaşmıştır (işlemci 1,261, bellek 1,314). Saatlik oranların dağılımı daha geniştir: fastStorage test döneminde saatlik oranın %95'lik yüzdeliği bellek için 1,96'dır, yani bazı saatlerde tepe talep ortalamanın yaklaşık iki katına çıkmıştır. Bu bulgu, planlamanın ortalama yerine tepe talebe göre yapılmasının gerekli olduğunu deneysel olarak göstermektedir."),
    tabloAlt("4.1", "Toplam saatlik tepe talebin toplam saatlik ortalama talebe oranı"),
    tabloG([2000, 1080, 1080, 1080, 1080, 1080, 1080],
      ["Grup", "Eğitim işlemci", "Eğitim bellek", "Doğrulama işlemci", "Doğrulama bellek", "Test işlemci", "Test bellek"], t41, { size: KUCUK }),
    not("Not: Alibaba için oran yalnızca eğitim döneminde hesaplanmıştır; bellek oranı r değerinden bağımsızdır."),
    P("Talebin işlemci ve bellek bileşenleri arasındaki oran Şekil 4.2'de, bu orana göre hangi kaynağın bağlayıcı olduğu ise Tablo 4.2'de verilmiştir. Bitbrains havuzlarında talep, çekirdek başına ortalama 1,2–1,5 GiB bellek gerektirmektedir. Bu oran, en düşük bellek oranlı hesaplama optimize ailenin (2 GB/vCPU) bile altındadır; dolayısıyla Bitbrains'te kapasiteyi çoğunlukla işlemci talebi belirler. Hesaplama optimize ailede bellek, fastStorage saatlerinin yalnızca %9,2'sinde, Rnd saatlerinin ise %28,5'inde bağlayıcı olmuştur. Bu nedenle ILP, Bitbrains'te büyük ölçüde c8i.xlarge tipini seçmiştir (Bölüm 4.3)."),
    ...sekil("sekiller/kaynak_orani.png", 1924, 1054, 14, "4.2", "Bitbrains havuzlarında bellek tepe talebinin işlemci tepe talebine oranının dağılımı (eğitim dönemi)"),
    P("Alibaba'da ise durum tersidir. Makineler bellek yoğun çalıştığından çekirdek başına bellek talebi r = 2 varsayımında bile 4,38 GB'tır ve bellek, hesaplama optimize ailede saatlerin tamamında bağlayıcıdır. r = 8 varsayımında talep çekirdek başına 17,5 GB'a çıkar ve en yüksek bellek oranlı aileyi (8 GB/vCPU) bile aşar. Bu durumda bellek kısıtını karşılamak için açılan sunucuların işlemci kapasitesinin büyük kısmı boşta kalır. Bu sonuç, Alibaba bulgularının M_{ref} varsayımına güçlü biçimde bağlı olduğunu göstermektedir."),
    tabloAlt("4.2", "Talebin bellek/işlemci oranı (GB/çekirdek) ve bellek kısıtının bağlayıcı olduğu saatlerin oranı (eğitim dönemi)"),
    tabloG([2000, 1100, 1000, 1100, 1100, 1100, 1100],
      ["Grup", "Oran medyan", "Oran %5", "Oran %95", "Bellek bağlayıcı, 2 GB/vCPU (%)", "Bellek bağlayıcı, 4 GB/vCPU (%)", "Bellek bağlayıcı, 8 GB/vCPU (%)"], t42, { size: KUCUK }),
    bosluk(),
    P("Mevsimsellik ve durağanlık analizleri, iki veri kümesinin zaman yapısının da farklı olduğunu göstermiştir. Bitbrains havuzlarında STL mevsimsellik gücü işlemci için 0,20–0,25, bellek için 0,16–0,22 aralığında kalmıştır (Tablo 4.4). Toplam serinin bir gün gecikmeli otokorelasyonu da fastStorage işlemci talebi için yalnızca 0,12'dir. Dolayısıyla Bitbrains'te günlük döngü zayıftır. Alibaba işlemci talebinde ise mevsimsellik gücü yaklaşık 0,83'tür ve belirgin bir günlük döngü vardır; Alibaba bellek talebinde bu değer 0,27–0,47 aralığındadır. Durağanlık testleri fastStorage serileri ve Alibaba işlemci serileri için d = 0, Rnd ve Alibaba bellek serileri için d = 1 sonucunu vermiştir."),

    // ===== 4.2
    H2("4.2 Tahmin Sonuçları"),
    P("Test dönemindeki gün-ileri tahmin ölçütleri Tablo 4.3'te verilmiştir. Alibaba'da işlemci ölçütleri r değerinden bağımsızdır. Bellek için ise MAE ve RMSE r ile orantılı olarak ölçeklenir, MAPE, MASE ve eksik tahmin oranı (ETO) değişmez; bu nedenle tabloda yalnızca r = 4 sonuçları verilmiştir."),
    tabloAlt("4.3", "Test dönemi gün-ileri tahmin ölçütleri"),
    tabloG([1500, 850, 1500, 1050, 1050, 900, 800, 850],
      ["Grup", "Kaynak", "Model", "MAE", "RMSE", "MAPE (%)", "MASE", "ETO (%)"], t43, { size: KUCUK, birlestir: 2 }),
    not("Not: MAE ve RMSE işlemci için çekirdek, bellek için GB cinsindendir."),
    P("Sonuçlar, tek bir modelin tüm havuz ve kaynaklarda en iyi olmadığını göstermektedir. Bitbrains fastStorage işlemci talebinde en düşük hatayı basit son değer modeli vermiştir (MAE 162,4, MASE 0,52). SARIMA bu modelin ardından gelmiş (MASE 0,72), LSTM ise mevsimsel naif modelin de gerisinde kalmıştır (MASE 1,00). Bellek talebinde sıralama değişmiş; SARIMA (MASE 1,04) ve LSTM (MASE 1,07) birbirine yakın sonuçlar vermiş, son değer modeli en kötü sonucu vermiştir (MASE 1,91). fastStorage alt havuzlarında da aynı tablo tekrarlanmıştır. Rnd havuzunda işlemci için SARIMA ve son değer modeli eşdeğer (MASE 0,62), bellek için ise SARIMA en iyidir (MASE 0,70). Bu sonuçlar, karmaşık modellerin basit modelleri her zaman geçemediği yönündeki literatür bulgularıyla uyumludur [@zeng]."),
    P("Ortalama hata ölçütleri tahminin yönü hakkında bilgi vermez. Eksik tahmin oranı bu açıdan farklı bir tablo ortaya koymuştur. fastStorage işlemci talebinde LSTM, test saatlerinin yalnızca %3,5 ± 1,2'sinde talebin altında tahmin yapmıştır. Bu oran SARIMA için %14,6, son değer modeli için %19,4, mevsimsel naif model için %46,5'tir. Başka bir deyişle LSTM bu havuzda sistematik olarak bir miktar yüksek tahmin yapmış, bu da ortalama hatasını artırırken eksik tahmini azaltmıştır (Şekil 4.3). Ancak bu özellik genellenebilir değildir: Rnd havuzunda LSTM'in eksik tahmin oranı %59,5, Alibaba işlemci talebinde %68,6'dır."),
    ...sekil("sekiller/sekil_4_1_tahmin_bitbrains.png", 1924, 1513, 15, "4.3", "Bitbrains fastStorage havuzunda test dönemi gerçek ve tahmin edilen saatlik tepe talep"),
    P("Alibaba'da tahmin başarımı belirgin biçimde daha düşüktür. İşlemci talebinde en iyi sonucu mevsimsel naif model vermiştir (MASE 1,14); bu, güçlü günlük döngüyle uyumludur. Tüm modellerde MASE 1'in üzerindedir. Bellek talebinde MAPE değerleri %2–6 gibi küçük değerlerdir, ancak eksik tahmin oranları %86–100 aralığındadır. Bu durum Şekil 4.4'te açıkça görülmektedir: test gününde bellek talebi, eğitim ve doğrulama dönemindeki düzeyin üzerine çıkmış ve tüm modeller bu düzey artışını kaçırmıştır. İşlemci talebi de test gününün son saatlerinde tüm tahminlerin üzerine çıkmıştır. Alibaba'da LSTM'in görece zayıf başarımında eğitim örneği sayısının azlığı da etkili olmuştur: 96 saatlik eğitim döneminden yalnızca 49 eğitim kökeni üretilebilmiştir."),
    ...sekil("sekiller/sekil_4_2_tahmin_alibaba.png", 1924, 1513, 15, "4.4", "Alibaba havuzunda (p0, r = 4) doğrulama ve test günlerinde gerçek ve tahmin edilen saatlik tepe talep"),
    P("Bitbrains havuzlarında seçilen SARIMA modelleri Tablo 4.4'te verilmiştir. Zayıf mevsimsellik nedeniyle hiçbir Bitbrains serisinde mevsimsel fark alınmamıştır (D = 0). Buna karşın 14 serinin 10'unda mevsimsel aile, doğrulama hatası bakımından mevsimsel olmayan aileden daha iyi bulunmuş ve seçilmiştir. Bu durum, günlük bileşenin zayıf da olsa tahmine katkı sağladığını göstermektedir. Alibaba havuzlarında seçilen modeller Ek B'de verilmiştir; Alibaba işlemci serilerinde D = 1 seçilmiştir."),
    tabloAlt("4.4", "Bitbrains havuzlarında seçilen SARIMA modelleri"),
    tabloG([1500, 850, 500, 500, 850, 1600, 2700],
      ["Havuz", "Kaynak", "d", "D", "STL F_{s}", "Seçilen aile", "Model"], t44, { size: KUCUK, birlestir: 1 }),
    bosluk(),

    // ===== 4.3
    H2("4.3 Kapasite Planlama Sonuçları"),
    P("Tahmine dayalı ILP planlarının ve kahin ILP planının test dönemindeki sonuçları Tablo 4.5'te verilmiştir. Regret, aynı havuzdaki kahin planına göre hesaplanmıştır."),
    tabloAlt("4.5", "Tahmine dayalı ILP ve kahin ILP sonuçları (test dönemi, α = 0,05, w = 0,25, L = 2)"),
    tabloG([1350, 1450, 900, 850, 850, 850, 850, 650, 800],
      ["Grup", "Plan", "Günlük maliyet (USD)", "Regret (%)", "Eksik saat (%)", "Karş. işlemci (%)", "Karş. bellek (%)", "En uzun olay (sa)", "Açılış"], t45, { size: KUCUK, birlestir: 1 }),
    bosluk(),
    P("Tablo 4.5'in en önemli bulgusu, tahmin doğruluğu ile karar kalitesi arasındaki ilişkinin doğrudan olmamasıdır. fastStorage'da en düşük tahmin hatasını veren son değer modeli 808 USD günlük maliyetle %19,4 eksik kapasiteli saat oranına yol açmıştır. En yüksek tahmin hatasını veren LSTM ise maliyeti kahin planına göre %60,1 artırmış, ancak eksik kapasiteli saat oranını %8,8'e ve karşılanmayan işlemci talebini %0,40'a indirmiştir. Mevsimsel naif model %15,7 ile en düşük regret değerini vermiş, ancak saatlerin %45,1'inde kapasite yetersiz kalmıştır. Dolayısıyla tahminin yalnızca ortalama hatası değil, talebin altında mı üstünde mi kaldığı da plan kalitesini belirlemektedir."),
    P("Rnd ve Alibaba havuzlarında negatif regret değerleri görülmektedir. Örneğin Rnd'de LSTM tabanlı plan kahin planından %12,7 daha ucuzdur; ancak saatlerin %55,6'sında kapasite yetersiz kalmıştır. Alibaba'da (r = 4) tüm tahmin modelleri negatif regret vermiş ve eksik kapasiteli saat oranları %38,5 ile %87,0 arasında gerçekleşmiştir. r = 2 varsayımında son değer modeliyle elde edilen plan kahinden %25,0 daha ucuzdur, ancak test saatlerinin %95,8'inde kapasite yetersizdir. Negatif regret bu nedenle bir başarı göstergesi değildir. ILP modeli gerçek talebi görmediğinden ve eksik kapasite için bir ceza terimi içermediğinden, talebin altında kalan bir tahmin doğrudan daha ucuz fakat yetersiz bir plana dönüşmektedir. Bu yanılsama, regret ölçütünün her zaman eksik kapasite ölçütleriyle birlikte okunması gerektiğini göstermektedir."),
    P("SARIMA tabanlı planın fastStorage havuzundaki görünümü Şekil 4.5'te verilmiştir. Bölüm 4.1'deki bulguyla uyumlu olarak plan neredeyse tamamen hesaplama optimize c8i.xlarge sunuculardan oluşmaktadır. İşlemci bağlayıcı kaynak olduğundan işlemci kapasitesi (1 + α) × tahmin çizgisiyle çakışmaktadır. Bellek kapasitesi ise tahmin edilen bellek talebinin üzerinde kalmıştır. Buna karşın eksik kapasite olaylarının bir bölümü, bellek talebinin kısa süreli sıçramalarından kaynaklanmıştır."),
    ...sekil("sekiller/sekil_4_3_ilp_plan.png", 1924, 1972, 13, "4.5", "Bitbrains fastStorage havuzunda SARIMA tahminine dayalı ILP planı: sunucu sayıları ile kapasite ve gerçek talep"),
    P("Tamsayılı modelin LP gevşetmesi ve yukarı yuvarlama sezgiseliyle karşılaştırması Tablo 4.6'da verilmiştir. Yuvarlanmış planların tamamı minimum çalışma süresi kısıtını sağlamıştır. Ortalama maliyet farkı küçüktür: Bitbrains'te %0,40, Alibaba'da %0,005. Ancak Bitbrains'te farkın en büyük değeri %12,9'a ulaşmıştır. Bu fark havuz büyüklüğüyle açıklanabilir. Alibaba havuzlarında aynı anda binlerce sunucu çalıştığından bir sunuculuk yuvarlama artığının etkisi önemsizdir. Bitbrains'te, özellikle 250 sanal makinelik alt havuzlarda, çalışan sunucu sayısı daha az olduğundan her yuvarlama artığı maliyete daha belirgin biçimde yansır. ILP modeli ortalama 0,02–0,04 saniyede, en kötü durumda 0,49 saniyede en iyi çözüme ulaşmıştır. Bu süreler, tam çözümün saatlik veya günlük planlamada pratik olarak uygulanabilir olduğunu göstermektedir; dolayısıyla yuvarlama sezgiseline başvurmak için hesaplama süresi açısından bir gerekçe yoktur."),
    tabloAlt("4.6", "ILP ile LP gevşetmesi ve yukarı yuvarlama çözümlerinin karşılaştırılması"),
    tabloG([1100, 900, 1100, 1100, 1100, 1100, 1050, 1050],
      ["Veri kümesi", "Çözüm sayısı", "Yuvarlanmış plan (3.27)'ye uygun (%)", "Maliyet farkı, ort. (%)", "Maliyet farkı, en büyük (%)", "LP alt sınır boşluğu, ort. (%)", "ILP süresi, ort. (s)", "ILP süresi, en büyük (s)"], t46, { size: KUCUK }),
    bosluk(),
    P("Güvenlik payı α'nın etkisi Tablo 4.7'de verilmiştir. α arttıkça maliyet doğrusal olarak artmış, eksik kapasiteli saat oranı ise azalmıştır. fastStorage'da α'nın 0'dan 0,20'ye çıkarılması günlük maliyeti 844 USD'den 1.013 USD'ye yükseltmiş, eksik kapasiteli saat oranını %18,8'den %8,3'e indirmiştir. Alibaba'da (r = 4) aynı değişiklik eksik kapasiteli saat oranını %96,4'ten %12,5'e indirmiştir; r = 8'de ise α ≥ 0,10 için eksik kapasite tamamen ortadan kalkmıştır. Kahin planı da aynı α ile planlandığından aynı α kahinine göre regret neredeyse sabit kalmıştır. Bu nedenle maliyet etkisi α = 0 kahinine göre regret ile değerlendirilmelidir."),
    tabloAlt("4.7", "Güvenlik payı α'nın SARIMA tabanlı ILP sonuçlarına etkisi (w = 0,25, L = 2)"),
    tabloG([1800, 700, 1200, 1300, 1300, 1100, 1100],
      ["Grup", "α", "Günlük maliyet (USD)", "Regret, aynı α kahinine göre (%)", "Regret, α = 0 kahinine göre (%)", "Eksik saat (%)", "En uzun olay (sa)"], t47, { size: KUCUK, birlestir: 1 }),
    bosluk(),
    P("Açma maliyeti katsayısı w ve minimum çalışma süresi L'nin etkisi Tablo 4.8'de verilmiştir. w arttıkça model, sunucuları sık açıp kapatmak yerine çalışır durumda tutmayı tercih etmiştir. fastStorage'da (L = 2) w = 0'da 438 olan açılış sayısı w = 1'de 218'e inmiştir. Bu sırada günlük maliyet yalnızca %1,2 artmış (884'ten 895 USD'ye), eksik kapasiteli saat oranı ise değişmemiştir. Alibaba'da (r = 4, w = 0) L'nin 1'den 4'e çıkarılması açılış sayısını 9.029'dan 5.667'ye indirmiştir. Bu sonuçlar, açma maliyeti ve minimum çalışma süresi kısıtlarının plan kararlılığını, maliyeti belirgin biçimde artırmadan sağladığını göstermektedir."),
    tabloAlt("4.8", "Açma maliyeti (w) ve minimum çalışma süresinin (L) SARIMA tabanlı ILP'de açılış sayısına etkisi (α = 0,05)"),
    tabloG([1800, 600, 1150, 1150, 1150, 1350, 1300],
      ["Grup", "w", "Açılış (L = 1)", "Açılış (L = 2)", "Açılış (L = 4)", "Günlük maliyet, L = 2 (USD)", "Eksik saat, L = 2 (%)"], t48, { size: KUCUK, birlestir: 1 }),
    bosluk(),

    // ===== 4.4
    H2("4.4 Otomatik Ölçeklendirme ile Karşılaştırma"),
    P("Stratejilerin Bitbrains ve Alibaba veri kümelerindeki karşılaştırması Tablo 4.9 ve Tablo 4.10'da verilmiştir. Tahmine dayalı ILP ve hibrit stratejiler bu tablolarda SARIMA tahminiyle gösterilmiştir; diğer tahmin modelleriyle elde edilen ILP sonuçları Tablo 4.5'tedir."),
    tabloAlt("4.9", "Bitbrains veri kümesinde stratejilerin karşılaştırılması (test dönemi)"),
    tabloG([1350, 1650, 850, 800, 800, 800, 800, 650, 800],
      ["Grup", "Strateji", "Günlük maliyet (USD)", "Regret (%)", "Eksik saat (%)", "Karş. işlemci (%)", "Karş. bellek (%)", "En uzun olay (sa)", "Açılış"], t49, { size: KUCUK, birlestir: 1 }),
    bosluk(),
    tabloAlt("4.10", "Alibaba veri kümesinde stratejilerin karşılaştırılması (test günü; M_{ref} varsayımına bağlı)"),
    tabloG([1350, 1650, 850, 800, 800, 800, 800, 650, 800],
      ["Grup", "Strateji", "Günlük maliyet (USD)", "Regret (%)", "Eksik saat (%)", "Karş. işlemci (%)", "Karş. bellek (%)", "En uzun olay (sa)", "Açılış"], t410, { size: KUCUK, birlestir: 1 }),
    bosluk(),
    P("R0 Tepkisel-ILP stratejisi tüm gruplarda kahin planına çok yakın bir maliyetle çalışmıştır (regret −0,8 ile 0,2 arasında). Ancak bir önceki saatin talebine göre karar verdiğinden talepteki her artışı bir saat gecikmeyle izlemiş; fastStorage'da saatlerin %31,9'unda, Alibaba'da (r = 4) %21,9'unda kapasite yetersiz kalmıştır. R1 eşik kuralı ise HPA'nın %70 hedef kullanımı nedeniyle yüksek bir boşta kapasiteyle çalışmış ve en pahalı stratejiler arasında yer almıştır (regret %54–104). Alibaba'da R1'in maliyeti, tek bir genel amaçlı sunucu tipiyle bellek talebinin karşılanmaya çalışılmasından dolayı ayrıca yükselmiştir."),
    P("Hibrit stratejinin iki varyantı, tahmine dayalı ILP'nin eksik kapasitesini belirgin biçimde azaltmıştır. Plana uyumlu eşikli R2, SARIMA tabanlı ILP'ye göre eksik kapasiteli saat oranını fastStorage'da %15,3'ten %8,3'e (1,8 kat), alt havuzlarda %15,6'dan %7,6'ya (2,1 kat), Rnd'de %33,3'ten %8,3'e (4,0 kat) indirmiştir. Alibaba'da bu azalma r = 2 için 3,9 kat (%58,9'dan %15,1'e), r = 4 için 5,0 kat (%41,7'den %8,3'e) ve r = 8 için 5,2 kat (%10,9'dan %2,1'e) olmuştur. En uzun eksik kapasite olayı fastStorage'da 7 saatten 3 saate, Rnd'de 16 saatten 2 saate, Alibaba'da (r = 4) yaklaşık 6 saatten 1,1 saate inmiştir. Bu iyileşmenin maliyeti, regret değerinde fastStorage'da 6,4, Alibaba'da (r = 4) 13,3 yüzde puanlık bir artıştır. HPA eşikli R2 eksik kapasiteyi daha da azaltmış (fastStorage'da %2,8, en uzun olay 1 saat), ancak tampon neredeyse sürekli devrede kaldığından maliyeti uyumlu varyanta göre çok daha yüksek olmuştur (fastStorage'da regret %68,1, Alibaba'da r = 4 için %50,3)."),
    P("Stratejilerin fastStorage havuzunda test döneminin ilk iki günündeki işlemci kapasiteleri Şekil 4.6'da görülmektedir. R0'ın kapasitesi gerçek talebi bir saat geriden izlemektedir. R1'in kapasitesi ise büyük adımlarla değişmektedir. R2 (uyumlu eşikler) çoğu saatte ILP planıyla çakışmakta, yalnızca gerçek talep plan kapasitesini aştığında devreye girmektedir; tamponun sıfır olduğu saatlerde R2 çizgisi ILP çizgisiyle üst üste düşmektedir."),
    ...sekil("sekiller/sekil_4_4_karsilastirma_zaman.png", 1924, 1156, 15, "4.6", "Bitbrains fastStorage havuzunda stratejilerin işlemci kapasitesi ve gerçek talep"),
    P("Şekil 4.7'de, güvenlik payı α değiştikçe stratejilerin regret (α = 0 kahinine göre) ve eksik kapasiteli saat oranı arasındaki dengesi gösterilmiştir. Eğrinin sol altına yakın noktalar hem ucuz hem de yeterli kapasiteli planları temsil eder. fastStorage havuzunda uyumlu eşikli R2 eğrisi, ILP eğrisinin solunda ve altında yer almaktadır. Örneğin %8,3'lük eksik kapasiteli saat oranına ILP ancak α = 0,20 ile ve %66,6 regret ile ulaşırken, R2 aynı orana α = 0,05 ile ve %52,5 regret ile ulaşmıştır. Rnd havuzunda R0, yüksek α değerlerinde R2 ile rekabet edebilir hâle gelmiştir. R1 tüm gruplarda diğer stratejilerin gerisinde kalmıştır. Alibaba'da eğriler birbirine yakındır ve ILP eğrisi α küçükken negatif regret bölgesine, yani yetersiz kapasite bölgesine kaymaktadır."),
    ...sekil("sekiller/sekil_4_5_maliyet_yetersizlik.png", 1924, 1902, 14, "4.7", "Maliyet–yetersizlik eğrisi: güvenlik payı α değiştikçe α = 0 kahinine göre regret ve eksik kapasiteli saat oranı (w = 0,25, L = 2)"),

    // ===== 4.5
    H2("4.5 Kontrollü Sentetik Senaryolar"),
    P("Sentetik senaryolarda stratejilerin regret ve eksik kapasiteli saat oranları Tablo 4.11'de, tahmin modellerinin MASE değerleri Tablo 4.12'de verilmiştir. Bu sonuçlar gerçek veri sonuçlarından ayrı tutulmuştur. Tablolardaki değerler Bitbrains'te 7, Alibaba'da 24 serinin (8 havuz × 3 r değeri) ortalamasıdır. Kontrol senaryosu, sentetik deney düzeneğinin ana deneylerin sonuçlarını yeniden ürettiğini doğrulamaktadır."),
    tabloAlt("4.11", "Sentetik senaryolarda stratejilerin regret ve eksik kapasiteli saat oranı (SENTETİK)"),
    tabloG([1900, 1900, 1200, 1150, 1200, 1150],
      ["Senaryo", "Strateji", "Bitbrains regret (%)", "Bitbrains eksik saat (%)", "Alibaba regret (%)", "Alibaba eksik saat (%)"], t411, { size: KUCUK, birlestir: 1 }),
    bosluk(),
    P("Ani talep değişimleri tahmine dayalı ILP'yi beklendiği gibi zorlamıştır. Basamak senaryosunda (talep × 1,5) ILP'nin eksik kapasiteli saat oranı Bitbrains'te %18,1'den %24,1'e, Alibaba'da %37,2'den %68,1'e yükselmiştir. Alibaba'da ILP'nin regret değeri %−26,8'e düşmüştür; bu, kahinin karşıladığı ek talebin ILP tarafından karşılanamamasının bir sonucudur. Uyumlu eşikli R2, tüm senaryolarda ILP'nin eksik kapasitesini önemli ölçüde azaltmıştır. Bitbrains'te R2'nin eksik kapasiteli saat oranı %6,1–10,6, Alibaba'da %3,6–11,5 aralığında kalmıştır. Ani düşüş senaryosunda ise durum tersine dönmüştür: talep yarıya indiğinde tahmine dayalı plan gereğinden büyük kalmış ve ILP'nin regret değeri Bitbrains'te %59,5'e, Alibaba'da %23,4'e çıkmıştır. R2'nin tamponu plandan küçük olamadığından bu fazlalığı gideremez. Bu senaryoda maliyet açısından en iyi strateji, tahmin kullanmayan R0 olmuştur."),
    P("Basamak ve darbe senaryolarındaki işlemci kapasiteleri Şekil 4.8'de görülmektedir. Darbe senaryosunda tepkisel tampon bir saatlik gecikmeyle devreye girebilmiş ve darbe sona erdikten sonra da minimum çalışma süresi ve soğuma süresi boyunca bir süre devrede kalmıştır. Ayrıca darbe, ardından başlayan günün tahminine de yansımış ve o günün ILP planını gereğinden yüksek tutmuştur. Bu gözlem, kısa süreli darbelerin hem tepkisel hem de tahmine dayalı yöntemler için zor olduğunu göstermektedir."),
    ...sekil("sekiller/sekil_4_6_sentetik_zaman.png", 1924, 1768, 14, "4.8", "Bitbrains fastStorage havuzunda basamak ve darbe senaryolarında işlemci kapasitesi ve bozulmuş talep (SENTETİK)"),
    P("Tablo 4.12'deki MASE değerleri, bozulmaların tahmin hatasına etkisini göstermektedir. Bitbrains'te bozulmalar MASE değerlerini sınırlı ölçüde artırmıştır; en büyük artış darbe senaryosunda görülmüştür (örneğin son değer modelinde bellek için 1,70'ten 2,55'e). Alibaba'da ise bellek talebinin düzeyi yüksek ve değişkenliği düşük olduğundan, basamak senaryosunda bellek MASE değerleri 20'nin üzerine çıkmıştır. Mevsimsellik bozulması senaryosunda Alibaba işlemci talebinin MASE değerleri 2,3–2,7'ye yükselmiştir. Bu, güçlü günlük döngüye dayanan modellerin günlük profil kaydığında belirgin biçimde bozulduğunu göstermektedir."),
    tabloAlt("4.12", "Sentetik senaryolarda test dönemi tahmin hatası, MASE (SENTETİK)"),
    tabloG([1900, 1900, 1200, 1150, 1200, 1150],
      ["Senaryo", "Model", "Bitbrains işlemci", "Bitbrains bellek", "Alibaba işlemci", "Alibaba bellek"], t412, { size: KUCUK, birlestir: 1 }),
    bosluk(),

    // ===== 4.6
    H2("4.6 Tartışma ve Sınırlılıklar"),
    P("Elde edilen bulgular dört ana noktada toplanabilir. Birincisi, tahmin doğruluğu ile karar kalitesi arasındaki ilişki doğrudan değildir. Ortalama hatası en düşük model her zaman en iyi planı vermemiş, tahminin yönü (eksik ya da fazla tahmin) plan kalitesinde belirleyici olmuştur. Bu durum, kapasite planlaması için tahmin modellerinin simetrik hata ölçütleri yerine asimetrik kayıp fonksiyonlarıyla veya yüksek yüzdelik (kantil) tahminleriyle eğitilmesinin daha uygun olabileceğini düşündürmektedir."),
    P("İkincisi, regret ölçütü tek başına yanıltıcıdır. ILP modeli gerçek talebi görmediğinden, talebin altında kalan tahminler kahinden daha ucuz fakat yetersiz planlar üretmiştir. Bu çalışmada regret her zaman eksik kapasite ölçütleriyle birlikte raporlanmış ve bu yanılsamanın etkisi açıkça gösterilmiştir."),
    P("Üçüncüsü, tamsayılı modelin çözümü hem kesin hem de hızlıdır. Ortalama çözüm süresi saniyenin onda birinin altındadır. LP gevşetmesi ve yuvarlama ortalamada en iyi çözüme çok yakın sonuç vermiş, ancak en kötü durumda %12,9 daha pahalı bir plan üretmiştir. Bu nedenle kesin çözümün tercih edilmesi için hesaplama maliyeti açısından bir engel bulunmamaktadır."),
    P("Dördüncüsü, proaktif ve tepkisel yaklaşımlar birbirinin zayıf yönlerini tamamlamaktadır. Plana uyumlu eşiklerle çalışan hibrit strateji, tahmine dayalı ILP'nin eksik kapasiteli saat oranını 1,8–5,2 kat azaltmış ve en uzun kapasite açığını en fazla 3 saate indirmiştir. HPA'nın varsayılan eşiklerinin ILP planıyla birlikte kullanılması ise eksik kapasiteyi daha da azaltmış, ancak maliyeti belirgin biçimde artırmıştır. Hangi varyantın tercih edileceği, uygulamanın eksik kapasiteye ne kadar tolerans gösterebildiğine bağlıdır."),
    P("Çalışmanın sınırlılıkları şunlardır:"),
    madde("(i) Alibaba veri kümesinin yalnızca 6 günü kullanılabilmiş ve test dönemi tek günle sınırlı kalmıştır. Alibaba sonuçları bu nedenle 24 saatlik tek bir gözleme dayanmaktadır ve test gününe özgü düzey artışından güçlü biçimde etkilenmiştir. Bu sonuçların genellenmesinde dikkatli olunmalıdır."),
    madde("(ii) Alibaba'da fiziksel bellek miktarı bilinmediğinden bellek talebi M_{ref} varsayımıyla hesaplanmıştır. Sonuçların r değerine güçlü biçimde bağlı olduğu görülmüştür."),
    madde("(iii) Kullanılan izler sanal makine ve fiziksel makine düzeyindedir ve Kubernetes düğüm havuzu talebi için vekil olarak kullanılmıştır. Pod'ların düğümlere yerleştirilmesi ve bunun yol açtığı parçalanma modellenmemiştir."),
    madde("(iv) Planlama saatlik çözünürlükte yapılmıştır ve sunucu açılış gecikmesi yalnızca bir saatlik karar gecikmesi ile açma maliyeti üzerinden temsil edilmiştir. Saat içi dinamikler ve uygulama düzeyindeki yanıt süreleri modellenmemiştir."),
    madde("(v) Maliyet yalnızca isteğe bağlı fiyatlarla hesaplanmıştır. Rezerve ve spot fiyatlandırma ile ağ ve depolama maliyetleri dikkate alınmamıştır."),
    madde("(vi) Günlük kayan pencerede her günün sonunda planın erken kapatılma eğilimi (ufuk sonu etkisi) düzeltilmemiştir.", true),
  ];
};
