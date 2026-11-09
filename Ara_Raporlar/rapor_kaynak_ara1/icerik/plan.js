// 5. SONRAKİ AŞAMALAR İÇİN ÇALIŞMA PLANI (1. ara rapor)
module.exports = (h) => {
  const { P, H1, tabloG, tabloAlt, bosluk } = h;
  const satirlar = [
    ["6", "Tahmin modellerinin test döneminde değerlendirilmesi (MAE, RMSE, MAPE, MASE, ETO) ve modellerin karşılaştırılması", "Soru (i)"],
    ["7", "ILP planlarının tüm havuzlarda kayan pencereyle üretilmesi; kahin ILP; LP gevşetmesi ve yuvarlama ile karşılaştırma", "Soru (ii), (iii)"],
    ["8", "Tepkisel (R0, R1) ve hibrit (R2) stratejilerin benzetimi ve tahmine dayalı ILP ile karşılaştırılması", "Soru (iv)"],
    ["9", "Duyarlılık analizi (α, w, L ve Alibaba için r) ve sentetik senaryo deneyleri", "Soru (ii), (iv)"],
    ["10", "Belirsizlik altında planlama yaklaşımlarının incelenmesi: kayan ufuklu yeniden planlama, kantil tahmine dayalı güvenlik payı ve iki aşamalı stokastik programlama", "Geliştirme"],
    ["11", "İkinci ara raporun hazırlanması ve teslimi", "2. ara rapor"],
    ["12", "Sonuçların istatistiksel anlamlılığının sınanması (Diebold–Mariano testi, bootstrap güven aralıkları) ve sunucu açılış gecikmesine duyarlılık", "Sağlamlık"],
    ["13", "Planın bir Kubernetes kümesinde uygulanmasını gösteren prototip (benzetimli düğümlerle)", "Uygulama"],
    ["14", "Ek geliştirmeler (rezerve ve spot fiyatlandırma, ek veri kümesi) ve sonuçların bütünleştirilmesi", "Geliştirme"],
    ["15", "Son raporun tamamlanması ve teslimi", "Son rapor"],
  ];
  return [
    H1("5. SONRAKİ AŞAMALAR İÇİN ÇALIŞMA PLANI"),
    P("Bu bölümde çalışmanın 6.–15. haftalarında yapılması planlanan işler verilmiştir. Plan, Bölüm 1.3'teki araştırma sorularını yanıtlamaya yönelik deneyleri, bu deneylerin sonuçlarına göre ele alınacak geliştirmeleri ve raporlama aşamalarını kapsamaktadır. Haftalık plan Tablo 5.1'de özetlenmiştir."),
    tabloAlt("5.1", "6.–15. haftalar için çalışma planı"),
    tabloG([900, 6100, 1700], ["Hafta", "Planlanan çalışma", "İlgili soru / çıktı"], satirlar, { size: 22, birlestir: 0, birlikte: true }),
    bosluk(),
    P("6.–9. haftalardaki deneyler, Bölüm 3'te tanımlanan yöntem ve ölçütlerle doğrudan araştırma sorularına yanıt verecektir. 10. haftada ele alınacak yaklaşımlar, sabit güvenlik payı α yerine talep belirsizliğini planlamaya daha doğrudan katmayı amaçlamaktadır. 12.–14. haftalardaki çalışmalar ise sonuçların sağlamlığını ve uygulanabilirliğini sınamaya yöneliktir; bu çalışmaların kapsamı temel deneylerin sonuçlarına göre önceliklendirilecektir."),
    P("İkinci ara rapor, bu raporun danışman geri bildirimleri doğrultusunda geliştirilmiş hâli olarak hazırlanacak ve ilk on haftada elde edilen sonuçları, karşılaşılan sorunları ve dönem sonuna kadar tamamlanacak işleri içerecektir."),
  ];
};
