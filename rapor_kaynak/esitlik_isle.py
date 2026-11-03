"""Derleme sonrası: docx içindeki §EQn§ yer tutucularını pandoc ile LaTeX'ten üretilen OMML eşitliklerle değiştirir.
Kullanım: python3 esitlik_isle.py rapor.docx esitlikler.json
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

docx, js = Path(sys.argv[1]), Path(sys.argv[2])
esitlikler = json.loads(js.read_text(encoding="utf-8"))
if not esitlikler:
    sys.exit(0)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # Her eşitlik ayrı bir görüntü bloğu; pandoc bunları sırayla oMathPara olarak üretir
    md = "\n\n".join(f"$${e['tex']}$$" for e in esitlikler)
    (td / "e.md").write_text(md, encoding="utf-8")
    subprocess.run(["pandoc", str(td / "e.md"), "-o", str(td / "e.docx")], check=True)
    with zipfile.ZipFile(td / "e.docx") as z:
        xml = z.read("word/document.xml").decode("utf-8")
    omml = re.findall(r"<m:oMathPara>.*?</m:oMathPara>", xml, flags=re.S)
    assert len(omml) == len(esitlikler), f"pandoc {len(omml)} eşitlik üretti, beklenen {len(esitlikler)}"
    # oMathPara içindeki oMath'ı al (satır içi kullanım), Times uyumlu yazı tipi yerine Cambria Math kalır
    omath = [re.search(r"<m:oMath>.*</m:oMath>", o, flags=re.S).group(0) for o in omml]

    # pandoc bazı özellik öğelerini OOXML şemasının beklediği sırada yazmıyor: dPr ve m:rPr çocuklarını sırala
    SIRA = {"m:dPr": ["m:begChr", "m:sepChr", "m:endChr", "m:grow", "m:shp", "m:ctrlPr"],
            "m:rPr": ["m:lit", "m:nor", "m:scr", "m:sty", "m:brk", "m:aln"]}

    def sirala(o: str) -> str:
        for etiket, sira in SIRA.items():
            def duzelt(mt, sira=sira):
                ic = mt.group(2)
                cocuklar = re.findall(r"<(m:[A-Za-z]+)\b[^>]*?/>|<(m:[A-Za-z]+)\b[^>]*>.*?</\2>", ic, flags=re.S)
                parcalar = [m.group(0) for m in re.finditer(r"<(m:[A-Za-z]+)\b[^>]*?/>|<(m:[A-Za-z]+)\b[^>]*>.*?</\2>", ic, flags=re.S)]
                adlar = [a or b for a, b in cocuklar]
                if "".join(parcalar).replace(" ", "") != ic.replace(" ", "").replace("\n", ""):
                    return mt.group(0)  # beklenmeyen içerik: dokunma
                sirali = sorted(zip(adlar, parcalar), key=lambda ab: sira.index(ab[0]) if ab[0] in sira else 99)
                return mt.group(1) + "".join(p for _, p in sirali) + "</" + etiket + ">"
            o = re.sub(r"(<" + etiket + r">)(.*?)</" + etiket + ">", duzelt, o, flags=re.S)
        return o
    omath = [sirala(o) for o in omath]

    with zipfile.ZipFile(docx) as z:
        dosyalar = {n: z.read(n) for n in z.namelist()}
    doc = dosyalar["word/document.xml"].decode("utf-8")
    sayac = 0
    for e, om in zip(esitlikler, omath):
        desen = re.compile(r"<w:r>(?:(?!</w:r>).)*?§EQ" + str(e["id"]) + r"§(?:(?!</w:r>).)*?</w:r>", re.S)
        doc, n = desen.subn(lambda _m, om=om: om, doc, count=1)
        assert n == 1, f"Yer tutucu bulunamadı: §EQ{e['id']}§ ({e['no']})"
        sayac += 1
    # Şekil/Tablo Listesi alanları stil adıyla toplanır (TOC \\t). Word bu anahtardaki ayırıcı olarak bölge ayarının
    # liste ayırıcısını kullanır; Türkçe Windows'ta bu ";" olduğundan "," yerine ";" yazılır.
    doc = doc.replace("Altyazisi,1&quot;", "Altyazisi;1&quot;")
    if 'xmlns:m=' not in doc[:3000]:
        doc = doc.replace("<w:document ", '<w:document xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" ', 1)
    dosyalar["word/document.xml"] = doc.encode("utf-8")
    gecici = td / "out.docx"
    with zipfile.ZipFile(gecici, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in dosyalar.items():
            z.writestr(n, b)
    shutil.copy(gecici, docx)
print(f"{sayac} eşitlik OMML'e çevrildi")
