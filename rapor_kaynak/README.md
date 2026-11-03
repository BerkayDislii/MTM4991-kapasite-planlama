# Rapor kaynakları (MTM4991 Tasarım Uygulaması)

`Tasarim_Uygulamasi_Rapor_Sablonu.docx` bu klasördeki betiklerle üretilir.

- `build.js` — docx-js ile raporu derler (DD-050 sayfa düzeni, başlıklar, şekil/tablo/eşitlik biçimleri).
- `esitlik_isle.py` — LaTeX eşitlikleri pandoc ile Word denklemlerine (OMML) çevirir.
- `toc_doldur.py` — İçindekiler, Şekil ve Tablo listelerini sayfa numaralarıyla önceden doldurur (LibreOffice gerekir).
- `icerik/` — bölüm metinleri: giris, literatur, metodoloji, bulgular, sonuc, onsayfa (Önsöz/Özet/Abstract), ekler; `refs.js` kaynakça.
- `veri/` — Bölüm 4 ve Ekler tablolarının CSV kaynakları (outputs/ klasöründen kopyalanır).
- `sekiller/` — rapordaki şekiller (Bölüm 3 şekilleri Graphviz `.dot` kaynaklarından üretilir).

Derleme: `node build.js ../Tasarim_Uygulamasi_Rapor_Sablonu.docx` (gereken: Node.js + `docx` paketi, Python 3, pandoc).
Kaynaklar metinde `[@anahtar]` biçiminde yazılır; numaralar ilk geçiş sırasına göre otomatik verilir.
