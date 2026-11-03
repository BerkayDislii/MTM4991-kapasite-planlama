// MTM4991 Tasarım Uygulaması rapor şablonu — YTÜ DD-050 kurallarına göre
const fs = require("fs");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel, Footer, PageNumber,
  NumberFormat, TableOfContents, Table, TableRow, TableCell, WidthType, BorderStyle,
  TabStopType, Math: OMath, MathRun, MathSum, MathSubScript, PageBreak, VerticalAlign,
  ShadingType,
} = d;

// ---- Ölçüler (DXA: 1 cm = 567) ----
const CM = 567;
const TEXT_W = Math.round(15 * CM); // 21 - 3,5 - 2,5 = 15 cm
const LINE = 240;                   // 1 satır (12 pt)
const GAP15 = 360;                  // 1,5 satır boşluk
const FONT = "Times New Roman";

// ---- Yardımcılar ----
// Basit işaretleme: a_{b} alt simge, a^{b} üst simge
// ---- Atıflar: [@a; @b] → [n, m]; numara ilk geçiş sırasına göre ----
const REFS = require("./icerik/refs.js");
const atifSirasi = [];
function atifCoz(text) {
  return text.replace(/\[@([^\]]+)\]/g, (_, ic) => {
    const nos = ic.split(";").map((k) => k.trim().replace(/^@/, "")).map((k) => {
      if (!REFS[k]) throw new Error("Bilinmeyen kaynak: " + k);
      if (!atifSirasi.includes(k)) atifSirasi.push(k);
      return atifSirasi.indexOf(k) + 1;
    });
    return "[" + nos.join(", ") + "]";
  });
}
function richRuns(text, base = {}) {
  text = atifCoz(text);
  const runs = [];
  const re = /([_^])\{([^}]*)\}/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    runs.push(new TextRun({ text: m[2], ...base, ...(m[1] === "_" ? { subScript: true } : { superScript: true }) }));
    last = re.lastIndex;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...base }));
  return runs;
}
const P = (text, opts = {}) => new Paragraph({ children: richRuns(text, opts.run || {}), ...opts.para });
// Yer tutucu: italik gri, köşeli parantez — son metinde silinecek
const PH = (text) => new Paragraph({ children: [new TextRun({ text: `[${text}]`, italics: true, color: "7F7F7F" })] });
const H1 = (text, opts = {}) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: richRuns(text), ...opts });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: richRuns(text) });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: richRuns(text) });
const blank = (n = 1) => Array.from({ length: n }, () => new Paragraph({ children: [], spacing: { after: 0 } }));
const center = (text, run = {}, after = 0) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after }, children: richRuns(text, run) });

// Şekil altyazısı: şeklin altında, 1 satır boşlukla, ortalı, koyu
const sekilAlt = (no, text) => new Paragraph({
  style: "SekilAltyazi",
  children: [new TextRun({ text: `Şekil ${no} `, bold: true }), ...richRuns(text, { bold: true })],
});
// Tablo altyazısı: tablonun üstünde, ortalı, tablo ile arasında 1 satır boşluk
const tabloAlt = (no, text) => new Paragraph({
  style: "TabloAltyazi",
  children: [new TextRun({ text: `Tablo ${no} `, bold: true }), ...richRuns(text)],
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, { bold = false, align = AlignmentType.CENTER } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, spacing: { after: 0 }, children: richRuns(text, { bold, size: 22 }) })],
  });
}
function table(widths, header, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { bold: true })) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], { align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })),
    ],
  });
}
// Gruplu tablo: ilk `birlestir` sütunda art arda aynı değerler dikey birleştirilir; size: yazı boyutu (yarım punto)
function tabloG(widths, header, rows, { size = 22, birlestir = 1, birlikte = rows.length <= 12 } = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  // birlikte: küçük tabloların sayfa arasında bölünmemesi için satırlar bir sonrakine bağlanır (keepNext)
  const hucre = (text, w, { bold = false, align = AlignmentType.CENTER, rowSpan, bagla = false } = {}) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, rowSpan,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ alignment: align, keepNext: bagla, spacing: { after: 0 }, children: richRuns(String(text), { bold, size }) })],
  });
  // rowSpan hesapla
  const span = rows.map((r) => r.map(() => 1));
  for (let c = 0; c < birlestir; c++) {
    for (let i = 0; i < rows.length; ) {
      let j = i + 1;
      while (j < rows.length && rows[j][c] === rows[i][c] && (c === 0 || span[j][c - 1] === 0 || rows[j][c - 1] === rows[i][c - 1])) j++;
      // üst sütun grubu sınırını aşma
      if (c > 0) { let k = i + 1; while (k < j && rows[k][c - 1] === rows[i][c - 1]) k++; j = k; }
      span[i][c] = j - i; for (let k = i + 1; k < j; k++) span[k][c] = 0;
      i = j;
    }
  }
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => hucre(h, widths[i], { bold: true, bagla: true })) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => span[ri][i] === 0 ? null :
        hucre(c, widths[i], { align: i < Math.max(1, birlestir) ? AlignmentType.LEFT : AlignmentType.CENTER,
          rowSpan: span[ri][i] > 1 ? span[ri][i] : undefined, bagla: birlikte && ri < rows.length - 1 })).filter(Boolean) })),
    ],
  });
}
// Liste satırı (Sembol / Kısaltma listesi): sol sütun + sekme + açıklama
const listeSatiri = (sol, sag) => new Paragraph({
  tabStops: [{ type: TabStopType.LEFT, position: 2268 }],
  indent: { left: 2268, hanging: 2268 },
  alignment: AlignmentType.LEFT,
  spacing: { after: 120 },
  children: [...richRuns(sol), new TextRun("\t"), ...richRuns(sag)],
});

// ---- Kapak bilgileri ----
const BASLIK = "KUBERNETES ORTAMINDA KAYNAK KULLANIMI TAHMİNİ VE TAMSAYILI DOĞRUSAL PROGRAMLAMA İLE MALİYET-OPTİMAL KAPASİTE PLANLAMASI";
function kapak() {
  return [
    ...blank(2),
    center("YILDIZ TEKNİK ÜNİVERSİTESİ", { bold: true }),
    center("KİMYA-METALÜRJİ FAKÜLTESİ", { bold: true }),
    center("MATEMATİK MÜHENDİSLİĞİ BÖLÜMÜ", { bold: true }),
    ...blank(5),
    center("MATEMATİK MÜHENDİSLİĞİNDE TASARIM UYGULAMASI", { bold: true }),
    ...blank(4),
    center(BASLIK, { bold: true, size: 28 }),
    ...blank(6),
    new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "Tez Yöneticisi: Doç. Dr. Kadriye ŞİMŞEK ALAN", bold: true })] }),
    ...blank(1),
    new Paragraph({
      tabStops: [{ type: TabStopType.LEFT, position: 3402 }], spacing: { after: 120 },
      children: [new TextRun({ text: "Öğrencinin Numarası\t: 22052063", bold: true })],
    }),
    new Paragraph({
      tabStops: [{ type: TabStopType.LEFT, position: 3402 }], spacing: { after: 120 },
      children: [new TextRun({ text: "Öğrencinin Adı Soyadı\t: Berkay DİŞLİ", bold: true })],
    }),
    ...blank(7),
    center("İstanbul, 2026", { bold: true }),
  ];
}

// ---- Sembol ve kısaltma listeleri (alfabetik; Latin harfler, ardından Yunan harfleri) ----
const SEMBOLLER = [
  ["B", "Geri kaydırma işleci (B y_{t} = y_{t−1})"],
  ["b_{t}", "t dilimindeki tampon sunucu sayısı (R1, R2)"],
  ["C_{k}", "k tipi sunucunun işlemci kapasitesi (vCPU)"],
  ["c_{k}", "k tipi sunucunun saatlik maliyeti (USD/saat)"],
  ["D^{5}_{τ}", "τ 5 dakikalık hücresindeki havuz toplam talebi"],
  ["D̄_{h}", "h saatindeki ortalama havuz talebi"],
  ["D_{h}", "h saatindeki saatlik tepe havuz talebi"],
  ["D^{cpu}_{t}, D^{mem}_{t}", "t dilimindeki gerçekleşen saatlik tepe işlemci (çekirdek) ve bellek (GB) talebi"],
  ["D̂^{cpu}_{t}, D̂^{mem}_{t}", "t dilimi için tahmin edilen saatlik tepe işlemci ve bellek talebi"],
  ["d_{k}", "k tipi bir sunucuyu kapatma maliyeti"],
  ["ESO", "Eksik kapasiteli saat oranı"],
  ["ETO", "Eksik tahmin oranı"],
  ["F_{s}", "STL ayrışımına dayalı mevsimsellik gücü"],
  ["g_{t}", "t diliminde gerekli tampon sunucu sayısı"],
  ["H", "Planlama ufku (saatlik dilim sayısı)"],
  ["h", "Saat indisi; tahmin adımı (Eşitlik 3.8–3.9)"],
  ["K", "Sunucu tipleri kümesi"],
  ["k", "Sunucu tipi indisi"],
  ["kap^{r}_{t}", "t dilimindeki r kaynağı kapasitesi (r ∈ {cpu, mem})"],
  ["L_{k}", "k tipi sunucunun minimum çalışma süresi (dilim)"],
  ["M_{k}", "k tipi sunucunun bellek kapasitesi (GiB)"],
  ["M_{ref}", "Alibaba makineleri için referans bellek miktarı (GB)"],
  ["N", "Test dönemindeki tahmin dilimi sayısı"],
  ["n", "Pod (kopya) sayısı"],
  ["o", "Tahmin kökeni (günün ilk saati)"],
  ["(p, d, q)×(P, D, Q)_{s}", "SARIMA modelinin mevsimsel olmayan ve mevsimsel dereceleri (s: mevsim uzunluğu)"],
  ["q^{r}", "Ölçek tipinin (m8i.xlarge) r kaynağındaki kapasitesi"],
  ["r", "Referans bellek katsayısı (GB/çekirdek); Bölüm 3.6–3.7'de kaynak indisi (cpu, mem)"],
  ["s_{k}", "k tipi bir sunucuyu açma (başlatma) maliyeti"],
  ["t", "Zaman dilimi (saat) indisi"],
  ["T", "Değerlendirme dönemindeki dilim sayısı"],
  ["u_{t}", "t dilimindeki kapasite kullanım oranı"],
  ["V", "Havuzdaki makinelerin kümesi"],
  ["w", "Açma maliyeti katsayısı (ısınma süresinin saat eşdeğeri)"],
  ["ŷ_{t}", "t dilimi için tahmin edilen değer"],
  ["x_{k,t}", "t diliminde çalışan k tipi sunucu sayısı"],
  ["x^{0}_{k}", "Planlama ufkunun başında çalışan k tipi sunucu sayısı"],
  ["y_{k,t}", "t diliminin başında açılan k tipi sunucu sayısı"],
  ["z_{k,t}", "t diliminin başında kapatılan k tipi sunucu sayısı"],
  ["α", "Güvenlik payı"],
  ["ε_{t}", "Beyaz gürültü (SARIMA hata terimi)"],
  ["ρ^{cpu}, ρ^{mem}", "Kullanılabilir işlemci ve bellek kapasitesi oranı"],
  ["σ", "Lojistik (sigmoid) fonksiyon"],
  ["τ_{üst}, τ_{hedef}, τ_{alt}", "Eşik kuralının üst, hedef ve alt kullanım eşikleri"],
  ["φ_{p}, θ_{q}, Φ_{P}, Θ_{Q}", "SARIMA modelinin otoregresif ve hareketli ortalama çok terimlileri"],
];
const KISALTMALAR = [
  ["ADF", "Augmented Dickey–Fuller (Genişletilmiş Dickey–Fuller Testi)"],
  ["AIC", "Akaike Information Criterion (Akaike Bilgi Ölçütü)"],
  ["ARIMA", "Autoregressive Integrated Moving Average (Otoregresif Bütünleşik Hareketli Ortalama)"],
  ["ARMA", "Autoregressive Moving Average (Otoregresif Hareketli Ortalama)"],
  ["AWS", "Amazon Web Services"],
  ["CP-SAT", "Constraint Programming – Satisfiability (Kısıt Programlama – Sağlanabilirlik Çözücüsü)"],
  ["CPU", "Central Processing Unit (Merkezi İşlem Birimi)"],
  ["EC2", "Elastic Compute Cloud"],
  ["GiB", "Gibibayt"],
  ["GLOP", "Google Linear Optimization Package (Google Doğrusal Optimizasyon Paketi)"],
  ["GPU", "Graphics Processing Unit (Grafik İşlem Birimi)"],
  ["HPA", "Horizontal Pod Autoscaler (Yatay Pod Otomatik Ölçekleyicisi)"],
  ["ILP", "Integer Linear Programming (Tamsayılı Doğrusal Programlama)"],
  ["KEDA", "Kubernetes Event-Driven Autoscaling (Kubernetes Olay Güdümlü Otomatik Ölçekleme)"],
  ["KPSS", "Kwiatkowski–Phillips–Schmidt–Shin Durağanlık Testi"],
  ["LP", "Linear Programming (Doğrusal Programlama)"],
  ["LSTM", "Long Short-Term Memory (Uzun Kısa Süreli Bellek)"],
  ["MAE", "Mean Absolute Error (Ortalama Mutlak Hata)"],
  ["MAPE", "Mean Absolute Percentage Error (Ortalama Mutlak Yüzde Hata)"],
  ["MASE", "Mean Absolute Scaled Error (Ortalama Mutlak Ölçeklenmiş Hata)"],
  ["MSE", "Mean Squared Error (Ortalama Kare Hata)"],
  ["RMSE", "Root Mean Square Error (Hata Kareleri Ortalamasının Karekökü)"],
  ["RNN", "Recurrent Neural Network (Tekrarlayan Sinir Ağı)"],
  ["SaaS", "Software as a Service (Hizmet Olarak Yazılım)"],
  ["SARIMA", "Seasonal Autoregressive Integrated Moving Average (Mevsimsel Otoregresif Bütünleşik Hareketli Ortalama)"],
  ["SCIP", "Solving Constraint Integer Programs"],
  ["SLA", "Service Level Agreement (Hizmet Düzeyi Anlaşması)"],
  ["STL", "Seasonal-Trend Decomposition using Loess (Loess ile Mevsim-Eğilim Ayrışımı)"],
  ["vCPU", "Virtual CPU (Sanal İşlemci)"],
  ["VM", "Virtual Machine (Sanal Makine)"],
  ["VPA", "Vertical Pod Autoscaler (Dikey Pod Otomatik Ölçekleyicisi)"],
];

// ---- Örnek eşitlik (3.1): amaç fonksiyonu; sol kenardan başlar, numara satır sonunda ----
const sub = (b, s) => new MathSubScript({ children: [new MathRun(b)], subScript: [new MathRun(s)] });
const nb = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const nbs = { top: nb, bottom: nb, left: nb, right: nb };
// ---- LaTeX eşitlik: yer tutucu yazılır, derleme sonrası esitlik_isle.py ile OMML'e çevrilir ----
const ESITLIKLER = [];
function eq(tex, no) {
  const id = ESITLIKLER.length;
  ESITLIKLER.push({ id, no, tex });
  const W1 = TEXT_W - 1134, W2 = 1134;
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [W1, W2], borders: { ...nbs, insideHorizontal: nb, insideVertical: nb },
    rows: [new TableRow({ cantSplit: true, children: [
      new TableCell({ borders: nbs, width: { size: W1, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, margins: { left: 0, right: 0 },
        children: [new Paragraph({ style: "Esitlik", children: [new TextRun(`§EQ${id}§`)] })] }),
      new TableCell({ borders: nbs, width: { size: W2, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER, margins: { left: 0, right: 0 },
        children: [new Paragraph({ style: "Esitlik", alignment: AlignmentType.RIGHT, children: [new TextRun(`(${no})`)] })] }),
    ] })],
  });
}
// Eşitlik + altında 1 satır boşluk (DD-050 3.6: metin ile eşitlik arasında 1 satır)
function eqB(tex, no) {
  return [eq(tex, no), new Paragraph({ spacing: { before: 0, after: 0 }, children: [] })];
}
// Şekil: ortalı görsel + altında altyazı. wCm: sayfadaki genişlik (cm)
function sekil(yol, wpx, hpx, wCm, no, baslik) {
  const w = Math.round(wCm / 2.54 * 96), hgt = Math.round(w * hpx / wpx);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 240, after: 0 },
      children: [new d.ImageRun({ type: "png", data: fs.readFileSync(yol), transformation: { width: w, height: hgt } })] }),
    sekilAlt(no, baslik),
  ];
}
const bosluk = () => new Paragraph({ spacing: { before: 0, after: 240 }, children: [] });
function esitlik(mathChildren, no) {
  const W1 = TEXT_W - 1134, W2 = 1134;
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [W1, W2], borders: { ...nbs, insideHorizontal: nb, insideVertical: nb },
    rows: [new TableRow({ children: [
      new TableCell({ borders: nbs, width: { size: W1, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
        margins: { left: 0, right: 0 },
        children: [new Paragraph({ style: "Esitlik", children: [new OMath({ children: mathChildren })] })] }),
      new TableCell({ borders: nbs, width: { size: W2, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
        margins: { left: 0, right: 0 },
        children: [new Paragraph({ style: "Esitlik", alignment: AlignmentType.RIGHT, children: [new TextRun(`(${no})`)] })] }),
    ] })],
  });
}
const m = {
  run: (t) => new MathRun(t),
  sub: (b, x) => new MathSubScript({ children: [new MathRun(b)], subScript: [new MathRun(x)] }),
  sup: (b, x) => new d.MathSuperScript({ children: [new MathRun(b)], superScript: [new MathRun(x)] }),
  frac: (num, den) => new d.MathFraction({ numerator: num, denominator: den }),
};


// ---- Gövde ----
const govde = [
  // 1. GİRİŞ (icerik/giris.js)
  ...require("./icerik/giris.js")({ P, H1, H2, H3, eq: eqB }).flat(),

  // 2. LİTERATÜR TARAMASI (icerik/literatur.js)
  ...require("./icerik/literatur.js")({ P, H1, H2, table, tabloAlt, Paragraph }),

  // 3. METODOLOJİ (icerik/metodoloji.js)
  ...require("./icerik/metodoloji.js")({ P, H1, H2, H3, eq: eqB, table, tabloAlt, bosluk,
    sekil: (...a) => sekil(...a) }).flat(),

  // 4. BULGULAR (icerik/bulgular.js)
  ...require("./icerik/bulgular.js")({ P, H1, H2, tabloG, tabloAlt, bosluk, sekil: (...a) => sekil(...a) }).flat(),

  // 5. SONUÇLAR VE ÖNERİLER (icerik/sonuc.js)
  ...require("./icerik/sonuc.js")({ P, H1, H2 }),

  // KAYNAKLAR
  H1("KAYNAKLAR"),
  ...(() => atifSirasi.map((k, i) => new Paragraph({
    alignment: AlignmentType.LEFT, indent: { left: 567, hanging: 567 }, spacing: { after: 120 },
    children: [new TextRun(`[${i + 1}]\t`), new TextRun(REFS[k])],
    tabStops: [{ type: TabStopType.LEFT, position: 567 }],
  })))(),

  // EKLER (icerik/ekler.js)
  ...require("./icerik/ekler.js")({ P, H1, H2, tabloG, tabloAlt, bosluk }),

  // ÖZGEÇMİŞ
  H1("ÖZGEÇMİŞ"),
  // DD-050 Ek 6 düzeni: sol sütun + sekme + bilgi
  ...[
    ["Ad Soyad", "Berkay DİŞLİ"],
    ["Doğum Tarihi", "07.09.2003"],
    ["Doğum Yeri", "Derince/Kocaeli"],
    ["Lise", "2017 – 2021 Özel İlk Karar Anadolu Lisesi"],
    ["Lisans", "2022 – Yıldız Teknik Üniversitesi, Matematik Mühendisliği Bölümü"],
    ["Staj Yaptığı Yerler", "Türkiye Halk Bankası A.Ş., Altyapı Yönetimi Daire Başkanlığı, Konteyner ve Yapay Zekâ Sistemleri Ekibi – İstanbul – (1 ay, 2026)"],
    ["", "LC Waikiki – Veri Bilimi – İstanbul – (1 ay, 2025)"],
  ].map(([a, b]) => new Paragraph({
    tabStops: [{ type: TabStopType.LEFT, position: 3118 }], indent: { left: 3118, hanging: 3118 },
    alignment: AlignmentType.LEFT, spacing: { after: 120 },
    children: [new TextRun(a), new TextRun("\t"), new TextRun(b)],
  })),
];

// ---- Ön sayfalar ----
const onSayfalar = [
  new Paragraph({ style: "OnBaslik", children: [new TextRun("İÇİNDEKİLER")] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 120 }, children: [new TextRun({ text: "Sayfa", bold: true })] }),
  new TableOfContents("İçindekiler", { hyperlink: true, headingStyleRange: "1-3" }),

  H1("SEMBOL LİSTESİ"),
  ...SEMBOLLER.map(([a, b]) => listeSatiri(a, b)),

  H1("KISALTMA LİSTESİ"),
  ...KISALTMALAR.map(([a, b]) => listeSatiri(a, b)),

  H1("ŞEKİL LİSTESİ"),
  new TableOfContents("Şekil Listesi", { hyperlink: true, stylesWithLevels: [{ styleName: "Sekil Altyazisi", level: 1 }] }),

  H1("TABLO LİSTESİ"),
  new TableOfContents("Tablo Listesi", { hyperlink: true, stylesWithLevels: [{ styleName: "Tablo Altyazisi", level: 1 }] }),

  ...require("./icerik/onsayfa.js")({ P, H1, Paragraph, TextRun, AlignmentType }),
];

// ---- Kapak bölümü (numarasız): dış kapak, iç kapak, telif sayfası ----
const kapakBolumu = [
  ...kapak(),
  new Paragraph({ children: [new PageBreak()] }),
  ...kapak(), // iç kapak
];
const telifBolumu = [
  // Telif notu: iç kapağın arka yüzünün en alt satırları (DD-050 2.2)
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
    frame: { type: "alignment", width: TEXT_W, height: 700,
      anchor: { horizontal: d.FrameAnchorType.MARGIN, vertical: d.FrameAnchorType.MARGIN },
      alignment: { x: d.HorizontalPositionAlign.CENTER, y: d.VerticalPositionAlign.BOTTOM } },
    children: [new TextRun({ text: "© Bu tezin bütün hakları Yıldız Teknik Üniversitesi Matematik Mühendisliği Bölümü'ne aittir.", size: 24 })] }),
];

const footer = () => new Footer({
  children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ children: [PageNumber.CURRENT] })] })],
});
const pageProps = (extra = {}) => ({
  page: {
    size: { width: 11906, height: 16838 },
    margin: { top: 3 * CM, bottom: 3 * CM, left: Math.round(3.5 * CM), right: Math.round(2.5 * CM), footer: Math.round(1.5 * CM) },
    ...extra,
  },
});

const doc = new Document({
  creator: "Berkay Dişli",
  title: "Kubernetes Ortamında Kaynak Kullanımı Tahmini ve Tamsayılı Doğrusal Programlama ile Maliyet-Optimal Kapasite Planlaması",
  description: "MTM4991 Matematik Mühendisliğinde Tasarım Uygulaması raporu (DD-050)",
  features: { updateFields: true },
  styles: {
    default: {
      document: {
        run: { font: FONT, size: 24 },
        paragraph: { spacing: { line: LINE, lineRule: d.LineRuleType.AUTO, after: GAP15 }, alignment: AlignmentType.JUSTIFIED },
      },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, allCaps: false },
        paragraph: { pageBreakBefore: true, spacing: { before: 0, after: GAP15 }, alignment: AlignmentType.LEFT, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 120, after: GAP15 }, alignment: AlignmentType.LEFT, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 120, after: GAP15 }, alignment: AlignmentType.LEFT, outlineLevel: 2, keepNext: true } },
      { id: "OnBaslik", name: "On Sayfa Basligi", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { after: GAP15 }, alignment: AlignmentType.LEFT } },
      { id: "SekilAltyazi", name: "Sekil Altyazisi", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: LINE, after: GAP15 }, alignment: AlignmentType.CENTER } },
      { id: "TabloAltyazi", name: "Tablo Altyazisi", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 24 },
        paragraph: { spacing: { before: GAP15, after: LINE }, alignment: AlignmentType.CENTER, keepNext: true } },
      { id: "Esitlik", name: "Esitlik", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 24 },
        paragraph: { spacing: { before: LINE, after: LINE }, alignment: AlignmentType.LEFT } },
    ],
  },
  sections: [
    { properties: pageProps(), children: kapakBolumu },
    { properties: { ...pageProps(), verticalAlign: VerticalAlign.BOTTOM }, children: telifBolumu },
    { properties: { ...pageProps(), page: { ...pageProps().page, pageNumbers: { start: 2, formatType: NumberFormat.LOWER_ROMAN } } },
      footers: { default: footer() }, children: onSayfalar },
    { properties: { ...pageProps(), page: { ...pageProps().page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      footers: { default: footer() }, children: govde },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const cikti = process.argv[2] || "sablon.docx";
  fs.writeFileSync(cikti, buf);
  fs.writeFileSync("esitlikler.json", JSON.stringify(ESITLIKLER));
  require("child_process").execSync(`python3 esitlik_isle.py ${cikti} esitlikler.json`, { stdio: "inherit" });
  require("child_process").execSync(`python3 toc_doldur.py ${cikti}`, { stdio: "inherit" });
  console.log("ok");
});
