// 4. İLK BEŞ HAFTADA TAMAMLANAN ÇALIŞMALAR VE ÖN SONUÇLAR (1. ara rapor)
const { csvOku, tr, ortStd } = require("./veri.js");

module.exports = (h) => {
  const { P, H1, H2, tabloG, tabloAlt, sekil, bosluk } = h;
  const madde = (t, son = false) => P(t, { para: { indent: { left: 567 }, spacing: { after: son ? 360 : 120 } } });
  const not = (t) => P(t, { run: { size: 20 }, para: { spacing: { before: 0, after: 360 } } });
  const KUCUK = 18;

  // Tablo 4.1: toplam tepe / toplam ortalama
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

  // Tablo 4.2: talep bileşimi ve bağlayıcı kaynak
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
    ["Alibaba (r = 2)", "4,38", "3,03", "6,20", "100,0", "61,5", "0,0"],
    ["Alibaba (r = 4)", "8,75", "6,06", "12,39", "100,0", "100,0", "61,5"],
    ["Alibaba (r = 8)", "17,50", "12,11", "24,79", "100,0", "100,0", "100,0"],
  ];

  return [
    H1("4. İLK BEŞ HAFTADA TAMAMLANAN ÇALIŞMALAR VE ÖN SONUÇLAR"),
    P("Bu bölümde çalışmanın ilk beş haftasında tamamlanan işler özetlenmiş, ardından veri kümelerinin keşifsel analizine ve ILP modelinin ilk uygulamasına ait ön sonuçlar verilmiştir. Bölümün sonunda bu aşamada karşılaşılan sorunlar ve bunlar için benimsenen çözüm yaklaşımları açıklanmıştır. Çalışma bireysel olarak yürütülmüş, tüm aşamalar rapor yazarı tarafından gerçekleştirilmiştir. Tablolarda ondalık ayırıcı olarak virgül kullanılmıştır."),

    // ===== 4.1
    H2("4.1 Tamamlanan Çalışmalar"),
    P("İlk beş haftada tamamlanan çalışmalar şunlardır:"),
    madde("(i) Konu ve kapsam: Çalışmanın konusu ve kapsamı danışman ile birlikte belirlenmiş, araştırma soruları tanımlanmıştır (Bölüm 1.3)."),
    madde("(ii) Literatür taraması: Bulut iş yükü tahmini, kapasite planlamasında tamsayılı programlama ve otomatik ölçekleme alanlarındaki çalışmalar incelenmiş ve bu çalışmanın bileşenleriyle karşılaştırılmıştır (Bölüm 2)."),
    madde("(iii) Veri edinimi ve ön işleme: Bitbrains GWA-T-12 ve Alibaba cluster-trace-v2018 izleri indirilmiş, veri kalitesi incelenmiştir. Zaman hizalama, birim dönüşümleri, eksik veri kuralları, havuz oluşturma ve saatlik tepe talep hesabı yazılım olarak gerçekleştirilmiş; 7 Bitbrains ve 24 Alibaba havuz serisi elde edilmiştir (Bölüm 3.2–3.3)."),
    madde("(iv) Keşifsel veri analizi: Tepe ve ortalama talep, talebin işlemci–bellek bileşimi, mevsimsellik ve durağanlık incelenmiştir (Bölüm 4.2)."),
    madde("(v) Tahmin modelleri: Mevsimsel naif ve son değer taban modelleri, SARIMA derece seçim yordamı ve LSTM modeli kodlanmıştır. SARIMA dereceleri ve LSTM eğitim ayarları yalnızca eğitim ve doğrulama verisi kullanılarak belirlenmiştir (Bölüm 3.4)."),
    madde("(vi) Sunucu tipleri ve fiyatlar: Planlamada kullanılacak üç AWS EC2 sunucu tipi seçilmiş, fiyatları AWS fiyat listesi programlama arayüzünden alınarak sabitlenmiştir (Tablo 3.3)."),
    madde("(vii) ILP modeli: Çok dönemli kapasite planlama modeli formüle edilmiş (Eşitlik 3.23–3.28) ve Google OR-Tools ile kodlanmıştır. Çözümleri denetleyen modelden bağımsız bir doğrulayıcı ile küçük örneklerde SCIP ve CP-SAT çözümlerini karşılaştıran birim testleri yazılmıştır (Bölüm 3.5, Ek A)."),
    madde("(viii) Karşılaştırma ve değerlendirme tasarımı: Kahin ILP, tepkisel ve hibrit stratejiler, değerlendirme ölçütleri, duyarlılık analizi ve sentetik senaryolar tanımlanmıştır (Bölüm 3.6–3.8).", true),

    // ===== 4.2
    H2("4.2 Keşifsel Veri Analizi"),
    P("Bitbrains fastStorage havuzunun saatlik tepe ve saatlik ortalama talebi Şekil 4.1'de görüldüğü gibidir. Talep, gün içi döngülerden çok birkaç gün süren düzey değişimleriyle karakterize edilir. İşlemci talebi eğitim döneminde yaklaşık 100 ile 1.200 çekirdek arasında değişmiş, test döneminde ise daha düşük bir düzeye inmiştir. Bellek talebinde kısa süreli ve yüksek sıçramalar dikkat çekmektedir. Test döneminde bu sıçramaların genliği eğitim dönemindekinden büyüktür; örneğin bellek tepe talebi test döneminde 2.500 GiB'ı aşmıştır. Test döneminin eğitim döneminden farklı bir davranış göstermesi, tahmin modellerini zorlayacak önemli bir etkendir."),
    ...sekil("sekiller/talep_zaman_bitbrains_fastStorage.png", 1924, 1564, 15, "4.1", "Bitbrains fastStorage havuzunun saatlik tepe ve saatlik ortalama işlemci ve bellek talebi"),
    P("Tepe talebin ortalama talebe oranı Tablo 4.1'de verilmiştir. Oran, her dönemde saatlik tepe değerlerin toplamının saatlik ortalama değerlerin toplamına bölünmesiyle hesaplanmıştır. Tüm havuz ve dönemlerde oran 1'den büyüktür; saat içi tepe, saatlik ortalamayı Bitbrains'te %7 ile %37, Alibaba'da işlemci için %17 ve bellek için %2 oranında aşmıştır. Oran, fastStorage havuzunun test döneminde en yüksek değerine ulaşmıştır (işlemci 1,261, bellek 1,314). Saatlik oranların dağılımı daha geniştir: fastStorage test döneminde saatlik oranın %95'lik yüzdeliği bellek için 1,96'dır, yani bazı saatlerde tepe talep ortalamanın yaklaşık iki katına çıkmıştır. Bu bulgu, planlamanın ortalama yerine tepe talebe göre yapılmasının gerekli olduğunu göstermektedir."),
    tabloAlt("4.1", "Toplam saatlik tepe talebin toplam saatlik ortalama talebe oranı"),
    tabloG([2000, 1080, 1080, 1080, 1080, 1080, 1080],
      ["Grup", "Eğitim işlemci", "Eğitim bellek", "Doğrulama işlemci", "Doğrulama bellek", "Test işlemci", "Test bellek"], t41, { size: KUCUK }),
    not("Not: Alibaba için oran yalnızca eğitim döneminde hesaplanmıştır; bellek oranı r değerinden bağımsızdır."),
    P("Talebin işlemci ve bellek bileşenleri arasındaki oran Şekil 4.2'de, bu orana göre hangi kaynağın bağlayıcı olduğu ise Tablo 4.2'de verilmiştir. Bitbrains havuzlarında talep, çekirdek başına ortalama 1,2–1,5 GiB bellek gerektirmektedir. Bu oran, en düşük bellek oranlı hesaplama optimize ailenin (2 GB/vCPU) bile altındadır; dolayısıyla Bitbrains'te kapasiteyi çoğunlukla işlemci talebi belirler. Hesaplama optimize ailede bellek, fastStorage saatlerinin yalnızca %9,2'sinde, Rnd saatlerinin ise %28,5'inde bağlayıcı olmuştur. Bu nedenle Bitbrains'te ILP'nin ağırlıklı olarak c8i.xlarge tipini seçmesi beklenir; Bölüm 4.3'teki ilk uygulama bu beklentiyi doğrulamıştır."),
    ...sekil("sekiller/kaynak_orani.png", 1924, 1054, 14, "4.2", "Bitbrains havuzlarında bellek tepe talebinin işlemci tepe talebine oranının dağılımı (eğitim dönemi)"),
    P("Alibaba'da ise durum tersidir. Makineler bellek yoğun çalıştığından çekirdek başına bellek talebi r = 2 varsayımında bile 4,38 GB'tır ve bellek, hesaplama optimize ailede saatlerin tamamında bağlayıcıdır. r = 8 varsayımında talep çekirdek başına 17,5 GB'a çıkar ve en yüksek bellek oranlı aileyi (8 GB/vCPU) bile aşar. Bu durumda bellek kısıtını karşılamak için açılan sunucuların işlemci kapasitesinin büyük kısmı boşta kalacaktır. Bu sonuç, Alibaba bulgularının M_{ref} varsayımına güçlü biçimde bağlı olacağını ve bu nedenle duyarlılık analizinin gerekli olduğunu göstermektedir."),
    tabloAlt("4.2", "Talebin bellek/işlemci oranı (GB/çekirdek) ve bellek kısıtının bağlayıcı olduğu saatlerin oranı (eğitim dönemi)"),
    tabloG([2000, 1100, 1000, 1100, 1100, 1100, 1100],
      ["Grup", "Oran medyan", "Oran %5", "Oran %95", "Bellek bağlayıcı, 2 GB/vCPU (%)", "Bellek bağlayıcı, 4 GB/vCPU (%)", "Bellek bağlayıcı, 8 GB/vCPU (%)"], t42, { size: KUCUK }),
    bosluk(),
    P("Mevsimsellik ve durağanlık analizleri, iki veri kümesinin zaman yapısının da farklı olduğunu göstermiştir. Bitbrains havuzlarında STL mevsimsellik gücü işlemci için 0,20–0,25, bellek için 0,16–0,22 aralığında kalmıştır. Toplam serinin bir gün gecikmeli otokorelasyonu da fastStorage işlemci talebi için yalnızca 0,12'dir. Dolayısıyla Bitbrains'te günlük döngü zayıftır. Alibaba işlemci talebinde ise mevsimsellik gücü yaklaşık 0,83'tür ve belirgin bir günlük döngü vardır; Alibaba bellek talebinde bu değer 0,27–0,47 aralığındadır. Durağanlık testleri fastStorage serileri ve Alibaba işlemci serileri için d = 0, Rnd ve Alibaba bellek serileri için d = 1 sonucunu vermiştir. Bu sonuçlar SARIMA derece seçiminde kullanılmıştır (Bölüm 3.4.3)."),

    // ===== 4.3
    H2("4.3 ILP Modelinin İlk Uygulaması"),
    P("Kurulan ILP modelinin doğru çalıştığını görmek amacıyla model, Bitbrains fastStorage havuzunda doğrulama döneminin ilk günü için SARIMA modelinin gün-ileri tahminleriyle çözülmüştür. Çözümde Tablo 3.4'teki merkez parametreler (α = 0,05, w = 0,25, L = 2) kullanılmış ve gün boş bir filoyla (x^{0} = 0) başlatılmıştır. Elde edilen plan Şekil 4.3'te görüldüğü gibidir."),
    ...sekil("sekiller/ornek_gunluk_plan.png", 1760, 1804, 13, "4.3", "Bitbrains fastStorage havuzunda bir günlük ILP planı: sunucu sayıları ile kapasite ve güvenlik paylı SARIMA tahmini"),
    P("SCIP çözücüsü problemi yaklaşık 0,06 saniyede en iyi çözüme ulaştırmış ve plan, modelden bağımsız doğrulayıcının tüm denetimlerinden geçmiştir. Planın açma maliyeti dahil günlük maliyeti 829,7 USD'dir. Plan yalnızca hesaplama optimize c8i.xlarge sunuculardan oluşmaktadır. İşlemci kapasitesi her saatte güvenlik paylı tahminin hemen üzerindedir; yani işlemci kısıtı bağlayıcıdır. Bellek kapasitesi ise tahminin belirgin biçimde üzerinde kalmıştır. Bu sonuç Bölüm 4.2'deki bulguyla uyumludur: talep çekirdek başına 2 GB'tan az bellek gerektirdiğinden en ucuz çözüm, vCPU başına en ucuz tip olan c8i.xlarge ile elde edilmektedir. Sunucu sayısının gün içinde 117'den 177'ye kademeli olarak artması, tahmin edilen işlemci talebindeki artışı izlemektedir."),
    P("Bu ilk uygulama, modelin kısıtları doğru biçimde temsil ettiğini ve günlük planlamada hesaplama süresinin bir sorun oluşturmayacağını göstermektedir. Planların gerçekleşen talep karşısındaki başarımının ölçülmesi ve diğer stratejilerle karşılaştırılması sonraki aşamada yapılacaktır (Bölüm 5)."),

    // ===== 4.4
    H2("4.4 Karşılaşılan Sorunlar ve Çözüm Yaklaşımları"),
    P("İlk beş haftada karşılaşılan başlıca sorunlar ve bunlar için benimsenen çözümler şunlardır:"),
    madde("(i) Alibaba izinin başındaki ölçüm boşlukları: İzin ilk yaklaşık 39 saatinde makinelerin önemli bir kısmı ölçüm göndermemiş, ardından küme genelinde bir ölçüm kesintisi yaşanmıştır. İlk iki gün analizden çıkarılarak uzun boşluk oranı %8,76'dan %0,15'e indirilmiştir (Bölüm 3.2.2)."),
    madde("(ii) Alibaba'da fiziksel bellek miktarının bilinmemesi: Bellek kullanımı yalnızca yüzde olarak verildiğinden, çekirdek başına bellek katsayısına dayanan bir referans bellek varsayımı tanımlanmış ve üç farklı değerle duyarlılık analizi yapılması kararlaştırılmıştır (Bölüm 3.3.3)."),
    madde("(iii) Zaman damgası biriminin belirsizliği: Zaman damgasının birimi sütun adına güvenilmeden, ardışık ölçümler arasındaki farkların medyanından belirlenmiştir (Bölüm 3.3.1)."),
    madde("(iv) Tepe talebin hesaplanma sırası: Makine bazında tepelerin toplanmasının talebi olduğundan yüksek gösterdiği belirlenmiş, tepe değer havuz toplamı üzerinden hesaplanmıştır (Bölüm 3.3.4)."),
    madde("(v) Alibaba'da LSTM için eğitim örneği azlığı: 96 saatlik eğitim dönemi 48 saatlik bakış penceresiyle yalnızca 25 eğitim örneği verdiğinden, bakış penceresi 24 saate indirilerek örnek sayısı 49'a çıkarılmıştır (Bölüm 3.4.4)."),
    madde("(vi) HPA eşiklerinin güvenlik paylı planla uyumsuzluğu: Varsayılan HPA eşiklerinin ILP planı üzerinde tamponu sürekli devrede tutacağı görülmüş, plana uyumlu eşikler tanımlanmıştır (Bölüm 3.6).", true),
  ];
};
