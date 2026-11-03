"""İçindekiler, Şekil Listesi ve Tablo Listesi alanlarını önceden doldurur.

Word bu alanları açılışta güncellemezse listeler boş görünür. Bu betik, alanların içine (önbellek sonucu olarak)
girdileri ve sayfa numaralarını yazar; Word'de F9 ile güncellendiğinde alanlar yeniden hesaplanır.
Sayfa numaraları LibreOffice ile PDF'e çevrilerek bulunur (Word'de bir sayfa kadar sapabilir; F9 düzeltir).
Kullanım: python3 toc_doldur.py rapor.docx
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

DOCX = Path(sys.argv[1])
METIN_GENISLIK = 8504  # DXA, 15 cm


def roma(n: int) -> str:
    t = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    s = ""
    for v, h in t:
        while n >= v:
            s, n = s + h, n - v
    return s


def paragraf_metni(p: str) -> str:
    import html
    return html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p)))


def oku(docx: Path) -> dict:
    with zipfile.ZipFile(docx) as z:
        return {n: z.read(n) for n in z.namelist()}


def yaz(docx: Path, dosyalar: dict) -> None:
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "o.docx"
        with zipfile.ZipFile(g, "w", zipfile.ZIP_DEFLATED) as z:
            for n, b in dosyalar.items():
                z.writestr(n, b)
        shutil.copy(g, docx)


def girdiler(doc: str) -> dict:
    """Gövde sırasıyla başlıklar ve altyazılar (sdt içindeki önbellek paragrafları hariç)."""
    govde = re.sub(r"<w:sdt>.*?</w:sdt>", "", doc, flags=re.S)
    icindekiler, sekil, tablo = [], [], []
    for p in re.findall(r"<w:p>.*?</w:p>|<w:p [^>]*>.*?</w:p>", govde, flags=re.S):
        st = re.search(r'<w:pStyle w:val="([^"]+)"/>', p)
        st = st.group(1) if st else ""
        m = paragraf_metni(p).strip()
        if not m:
            continue
        if st in ("Heading1", "Heading2", "Heading3"):
            icindekiler.append((int(st[-1]), m))
        elif st == "SekilAltyazi":
            sekil.append((1, m))
        elif st == "TabloAltyazi":
            tablo.append((1, m))
    return {"İçindekiler": icindekiler, "Şekil Listesi": sekil, "Tablo Listesi": tablo}


def sayfalar(docx: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", td, str(docx)],
                       check=True, capture_output=True)
        pdf = Path(td) / (docx.stem + ".pdf")
        metin = subprocess.run(["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
    return metin.split("\f")


def normal(s: str) -> str:
    return re.sub(r"\s+", "", s)


def govde_basi(ns: list[str]) -> int:
    return next(i for i, s in enumerate(ns) if s.startswith("1.GİRİŞ"))


def etiketle(i: int, govde_bas: int) -> str:
    # 0 tabanlı PDF sayfası; ön sayfalar 4. PDF sayfasında (indis 3) "ii" ile başlar
    return str(i - govde_bas + 1) if i >= govde_bas else roma(i - 3 + 2)


def sayfa_bul(girdi_listesi: dict, bos: list[str], dolu: list[str] | None) -> dict:
    """Gövde girdileri boş listeli çıktıda aranır (liste metinleri karışmaz, arap numaralar ön sayfa uzunluğundan
    bağımsızdır). Ön sayfa başlıkları (roma) dolu listeli çıktıda, o başlıkla başlayan sayfa olarak bulunur."""
    nb = [normal(x) for x in bos]
    gb = govde_basi(nb)
    nd = [normal(x) for x in dolu] if dolu else nb
    gd = govde_basi(nd)
    sonuc = {}
    for ad, liste in girdi_listesi.items():
        k, cikti = gb, []
        for duzey, metin in liste:
            hedef = normal(metin)
            on_sayfa = not re.match(r"^(\d|EK|KAYNAKLAR|EKLER|ÖZGEÇMİŞ|Ek )", metin)
            if ad == "İçindekiler" and on_sayfa:
                j = next((i for i in range(3, gd) if nd[i].startswith(hedef)), 3)
                cikti.append((duzey, metin, etiketle(j, gd)))
                continue
            j = next((i for i in range(k, len(nb)) if hedef[:40] in nb[i]), None)
            if j is None:
                j = next((i for i in range(k, len(nb)) if hedef[:15] in nb[i]), k)
            cikti.append((duzey, metin, etiketle(j, gb)))
            k = j
        sonuc[ad] = cikti
    return sonuc


def girdi_xml(duzey: int, metin: str, sayfa: str) -> str:
    sol = {1: 0, 2: 284, 3: 567}[duzey]
    kalin = "<w:b/>" if duzey == 1 and False else ""
    return ('<w:p><w:pPr><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="%d"/></w:tabs>'
            '<w:spacing w:before="0" w:after="60"/><w:ind w:left="%d" w:right="567"/><w:jc w:val="left"/></w:pPr>'
            '<w:r><w:rPr>%s</w:rPr><w:t xml:space="preserve">%s</w:t></w:r><w:r><w:tab/></w:r>'
            '<w:r><w:t>%s</w:t></w:r></w:p>') % (METIN_GENISLIK, sol, kalin, escape(metin), sayfa)


def doldur(doc: str, sonuc: dict) -> str:
    for ad, liste in sonuc.items():
        desen = re.compile(r'(<w:sdt><w:sdtPr><w:alias w:val="' + re.escape(ad) + r'"/></w:sdtPr><w:sdtContent>'
                           r'<w:p><w:r><w:fldChar w:fldCharType="begin"[^>]*/><w:instrText[^>]*>[^<]*</w:instrText>'
                           r'<w:fldChar w:fldCharType="separate"/></w:r></w:p>)(.*?)'
                           r'(<w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:sdtContent></w:sdt>)', re.S)
        icerik = "".join(girdi_xml(*g) for g in liste)
        doc, n = desen.subn(lambda m: m.group(1) + icerik + m.group(3), doc, count=1)
        assert n == 1, f"{ad} alanı bulunamadı"
    return doc


dosyalar = oku(DOCX)
doc0 = dosyalar["word/document.xml"].decode("utf-8")
liste = girdiler(doc0)
bos = sayfalar(DOCX)
dolu, onceki = None, None
for tur in range(4):  # ön sayfa uzunluğu değiştikçe roma numaraları kayar: sabitlenene kadar tekrar
    sonuc = sayfa_bul(liste, bos, dolu)
    if sonuc == onceki:
        break
    dosyalar["word/document.xml"] = doldur(doc0, sonuc).encode("utf-8")
    yaz(DOCX, dosyalar)
    dolu, onceki = sayfalar(DOCX), sonuc
print("Listeler dolduruldu: " + ", ".join(f"{k} {len(v)}" for k, v in sonuc.items()))
