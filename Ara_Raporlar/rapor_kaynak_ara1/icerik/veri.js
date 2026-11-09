// CSV okuma ve biçimlendirme yardımcıları (Bölüm 4 tabloları doğrudan proje çıktılarından üretilir)
const fs = require("fs");
const path = require("path");
const KLASOR = path.join(__dirname, "..", "veri");

function csvOku(ad) {
  const metin = fs.readFileSync(path.join(KLASOR, ad), "utf8").replace(/^﻿/, "").trim();
  const satirlar = [];
  for (const satir of metin.split(/\r?\n/)) {
    const alan = []; let cur = "", q = false;
    for (let i = 0; i < satir.length; i++) {
      const ch = satir[i];
      if (q) { if (ch === '"' && satir[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false; else cur += ch; }
      else if (ch === '"') q = true; else if (ch === ",") { alan.push(cur); cur = ""; } else cur += ch;
    }
    alan.push(cur); satirlar.push(alan);
  }
  const [bas, ...gov] = satirlar;
  return gov.map((r) => Object.fromEntries(bas.map((b, i) => [b, r[i]])));
}
// Sayıyı Türkçe ondalık virgülle yazar
const tr = (x, n = 2) => Number(x).toFixed(n).replace(".", ",");
const ortStd = (a, n = 2) => {
  const m = a.reduce((s, v) => s + v, 0) / a.length;
  const sd = Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
  return `${tr(m, n)} ± ${tr(sd, n)}`;
};
module.exports = { csvOku, tr, ortStd };
