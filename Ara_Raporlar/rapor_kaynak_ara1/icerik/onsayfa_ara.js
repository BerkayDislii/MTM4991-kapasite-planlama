// Ön sayfa: ÖZET (1. ara rapor)
module.exports = (h) => {
  const { P, H1, Paragraph, TextRun } = h;
  const anahtar = (bas, metin) => new Paragraph({ children: [new TextRun({ text: bas, bold: true }), new TextRun(metin)] });
  return [
    H1("ÖZET"),
    P("Bulut ortamlarında çalışan uygulamalar için ayrılan kapasitenin talebin üzerinde kalması gereksiz maliyete, altında kalması ise hizmet düzeyi ihlallerine yol açar. Bu çalışmada, bir Kubernetes düğüm havuzunun işlemci ve bellek talebini bir gün öncesinden tahmin eden ve bu tahmine göre maliyeti en küçük sunucu planını tamsayılı doğrusal programlama (ILP) ile belirleyen bir kapasite planlama sistemi tasarlanmaktadır. Bu rapor, çalışmanın ilk beş haftasını kapsayan birinci ara rapordur."),
    P("İlk beş haftada çalışmanın kapsamı belirlenmiş; bulut iş yükü tahmini, kapasite planlamasında tamsayılı programlama ve otomatik ölçekleme alanlarında literatür taraması yapılmıştır. Bitbrains GWA-T-12 ve Alibaba cluster-trace-v2018 veri kümeleri 5 dakikalık ortak bir ızgaraya hizalanmış ve havuz toplamının saatlik tepesi planlama talebi olarak tanımlanmıştır. Tahmin için mevsimsel naif ve son değer taban modelleri, SARIMA ve LSTM modelleri kurulmuştur. Üç AWS EC2 sunucu tipi üzerinden, açma maliyeti ve minimum çalışma süresi kısıtları içeren çok dönemli bir ILP modeli formüle edilmiş ve Google OR-Tools aracılığıyla SCIP çözücüsüyle çözülecek biçimde kodlanmıştır."),
    P("Keşifsel analizde saat içi tepe talebin saatlik ortalama talebi 1,02–1,37 kat aştığı ve bu nedenle planlamanın tepe talebe göre yapılması gerektiği görülmüştür. Bitbrains havuzlarında işlemcinin, Alibaba havuzlarında ise belleğin bağlayıcı kaynak olduğu belirlenmiştir. ILP modelinin ilk uygulamasında bir günlük plan 0,1 saniyenin altında en iyi çözüme ulaşmıştır. Sonraki aşamalarda tahmin ve planlama sonuçlarının test döneminde değerlendirilmesi, tepkisel ve hibrit stratejilerle karşılaştırma, duyarlılık analizi ve sentetik senaryo deneyleri yapılacaktır."),
    anahtar("Anahtar Kelimeler: ", "Kapasite planlama, tamsayılı doğrusal programlama, zaman serisi tahmini, Kubernetes, otomatik ölçekleme, bulut bilişim."),
  ];
};
