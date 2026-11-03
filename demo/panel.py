"""Canlı panel: demonun yazdığı CSV'leri 2 saniyede bir okuyup çizer; senaryo düğmeleriyle talebi bozar.

Çalıştırma (proje kökünden, demo çalışırken ayrı bir terminalde):
    .venv/bin/streamlit run demo/panel.py --client.toolbarMode minimal --browser.gatherUsageStats false
Tarayıcıda http://localhost:8501 açılır.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

KOK = Path(__file__).resolve().parents[1]
with open(Path(__file__).with_name("ayar.yaml"), encoding="utf-8") as f:
    CIKTI = KOK / yaml.safe_load(f)["cikti"]
with open(KOK / "configs" / "synthetic.yaml", encoding="utf-8") as f:
    SENARYO_CFG = yaml.safe_load(f)["senaryolar"]

PENCERE_SAAT = 48                                   # grafiklerde gösterilen son saat sayısı
AD = {"ilp": "Yalnız ILP", "r2_uyumlu": "ILP + tampon"}
RENK = {"Talep": "#2A6FC4", "Yalnız ILP": "#8A94A6", "ILP + tampon": "#C2501F"}
SENARYO_AD = {"darbe": "Darbe", "basamak": "Basamak", "dusus": "Düşüş"}
SENARYO_ACIKLAMA = {
    "darbe": "Talep {sure} saat boyunca ×{c:g}: beklenmedik kısa bir yoğunluk (ör. kampanya, haber).",
    "basamak": "Talep kalıcı olarak ×{c:g}: yeni bir müşteri ya da servis devreye girdi.",
    "dusus": "Talep kalıcı olarak ×{c:g}: bir servis kapandı ya da başka bölgeye taşındı.",
}

st.set_page_config(page_title="Kapasite Planlayıcı", layout="wide")
st.markdown("<style>div.block-container{padding-top:1.5rem} [data-testid='stMetricValue']{font-size:1.7rem}"
            "</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ veri

def oku() -> tuple[dict | None, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """durum.json ve her grubun adim.csv / saat.csv dosyaları (grup → tablo)."""
    try:
        meta = json.loads((CIKTI / "durum.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, {}, {}
    adim, saat = {}, {}
    for kol in meta.get("kollar", []):
        st_ad = kol["strateji"]
        try:
            adim[st_ad] = pd.read_csv(CIKTI / kol["grup"] / "adim.csv")
            saat[st_ad] = pd.read_csv(CIKTI / kol["grup"] / "saat.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
    return meta, adim, saat


def komut_gonder(senaryo: str) -> None:
    gecici = CIKTI / "komut.json.tmp"
    gecici.write_text(json.dumps({"senaryo": senaryo, "id": time.time()}), encoding="utf-8")
    gecici.replace(CIKTI / "komut.json")


# ------------------------------------------------------------------ grafikler

def eksen(alt_sinir: float, ust_sinir: float) -> alt.X:
    return alt.X("t:Q", title="Benzetim saati", scale=alt.Scale(domain=[alt_sinir, ust_sinir], nice=False))


def senaryo_bantlari(meta: dict, bas: int, x0: float, x1: float, simdi: float) -> alt.Chart | None:
    """Senaryonun etkin olduğu aralıklar (açık turuncu bant)."""
    satir = []
    for g in meta.get("senaryo", {}).get("gecmis", []):
        b = g["bas"] / 12 - bas
        e = (g["bit"] / 12 - bas) if g["bit"] is not None else simdi
        e = min(e, simdi)
        if e > x0 and b < x1:
            satir.append({"b": max(b, x0), "e": min(e, x1), "Senaryo": SENARYO_AD[g["ad"]]})
    if not satir:
        return None
    return alt.Chart(pd.DataFrame(satir)).mark_rect(color="#F2C9A8", opacity=0.45).encode(
        x="b:Q", x2="e:Q", tooltip=["Senaryo:N"])


def katman(grafik: alt.Chart, bant: alt.Chart | None) -> alt.Chart:
    return (bant + grafik) if bant is not None else grafik


def kapasite_grafigi(adim: dict, kaynak: str, birim: str, x: alt.X, bant) -> alt.Chart:
    """Talep ve iki stratejinin kapasitesi aynı grafikte: aradaki fark tamponun etkisi."""
    parca = []
    ilk = next(iter(adim.values()))
    parca.append(pd.DataFrame({"t": ilk["t"], "v": ilk[f"talep_{kaynak}"], "Seri": "Talep"}))
    for st_ad, a in adim.items():
        parca.append(pd.DataFrame({"t": a["t"], "v": a[f"kap_{kaynak}"], "Seri": AD[st_ad]}))
    d = pd.concat(parca)
    seriler = [s for s in RENK if s in set(d["Seri"])]
    cizgi = alt.Chart(d).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=x, y=alt.Y("v:Q", title=birim),
        color=alt.Color("Seri:N", scale=alt.Scale(domain=seriler, range=[RENK[s] for s in seriler]),
                        legend=alt.Legend(orient="top", title=None)))
    return katman(cizgi, bant).properties(height=250)


def bekleyen_grafigi(adim: dict, x: alt.X, bant) -> alt.Chart:
    d = pd.concat([pd.DataFrame({"t": a["t"], "Pod": a["bekleyen_cpu"] + a["bekleyen_mem"], "Seri": AD[s]})
                   for s, a in adim.items()])
    seriler = [s for s in RENK if s in set(d["Seri"])]
    cizgi = alt.Chart(d).mark_area(interpolate="step-after", opacity=0.75).encode(
        x=x, y=alt.Y("Pod:Q", stack=None, title="Yer bulamayan pod"),
        color=alt.Color("Seri:N", scale=alt.Scale(domain=seriler, range=[RENK[s] for s in seriler]),
                        legend=alt.Legend(orient="top", title=None)))
    return katman(cizgi, bant).properties(height=250)


def dugum_grafigi(saat: dict, x: alt.X, bant) -> alt.Chart:
    parca = []
    for s, a in saat.items():
        tip = [c for c in a.columns if c.startswith("x_")]
        b = a.assign(Düğüm=a[tip].sum(1))
        b = pd.concat([b, b.iloc[[-1]].assign(t=b["t"].iloc[-1] + 1)])     # son saati tam genişlikte çiz
        parca.append(pd.DataFrame({"t": b["t"], "Düğüm": b["Düğüm"], "Seri": AD[s]}))
    d = pd.concat(parca)
    seriler = [s for s in RENK if s in set(d["Seri"])]
    cizgi = alt.Chart(d).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=x, y=alt.Y("Düğüm:Q", title="Çalışan düğüm", scale=alt.Scale(zero=False)),
        color=alt.Color("Seri:N", scale=alt.Scale(domain=seriler, range=[RENK[s] for s in seriler]),
                        legend=alt.Legend(orient="top", title=None)))
    return katman(cizgi, bant).properties(height=250)


# ------------------------------------------------------------------ sayfa

def senaryo_paneli(meta: dict | None) -> None:
    etkin = (meta or {}).get("senaryo", {}).get("etkin", "")
    calisiyor = meta is not None and not meta.get("bitti", True)
    st.markdown("**Senaryo: talebi canlı boz**" + (f" · şu an etkin: **{SENARYO_AD[etkin]}**" if etkin else ""))
    k = st.columns([1, 1, 1, 1, 3])
    for i, ad in enumerate(SENARYO_AD):
        sc = SENARYO_CFG[ad]
        if k[i].button(SENARYO_AD[ad], use_container_width=True, disabled=not calisiyor,
                       help=SENARYO_ACIKLAMA[ad].format(sure=sc.get("sure_saat"), c=sc["carpan"])):
            komut_gonder(ad)
    if k[3].button("Normale dön", use_container_width=True, disabled=not (calisiyor and etkin)):
        komut_gonder("normal")
    k[4].caption("Çarpanlar raporun sentetik senaryolarıyla aynı (configs/synthetic.yaml). Senaryo sonraki 5 dk'lık "
                 "adımda başlar; sabah yapılan ILP planı bundan habersizdir.")


@st.fragment(run_every="2s")
def canli() -> None:
    meta, adim, saat = oku()
    senaryo_paneli(meta)
    if meta is None or not adim or any(a.empty for a in adim.values()):
        st.info("Demo henüz başlamadı. Başka bir terminalde: `.venv/bin/python -m demo.calistir`")
        return
    bas = meta["bas"]
    for a in adim.values():
        a["t"] = a["slot"] - bas + a["adim"] / 12
    for s in saat.values():
        s["t"] = s["slot"] - bas
    ilk = next(iter(adim.values()))
    son = ilk.iloc[-1]
    simdi = float(ilk["t"].max()) + 1 / 12

    st.caption(f"{meta['havuz']} · tahmin: {meta['model'].upper()} · Gün {int(son['gun'])}, "
               f"{int(son['saat']):02d}:{int(son['adim']) * 5:02d} · {'tamamlandı' if meta['bitti'] else 'çalışıyor'}")

    # Karşılaştırma tablosu: her strateji bir sütun
    kolon = st.columns(len(adim))
    for c, (st_ad, a) in zip(kolon, adim.items()):
        s = saat.get(st_ad)
        with c:
            st.markdown(f"<span style='color:{RENK[AD[st_ad]]};font-weight:700;font-size:1.15rem'>"
                        f"■ {AD[st_ad]}</span>", unsafe_allow_html=True)
            m = st.columns(4)
            m[0].metric("Yer bulamayan pod", int(a.iloc[-1]["bekleyen_cpu"] + a.iloc[-1]["bekleyen_mem"]))
            if s is not None and not s.empty:
                n_eksik, n = int(s["eksik"].sum()), len(s)
                m[1].metric("Eksik saat", f"{n_eksik} / {n}")
                m[2].metric("Çalışan düğüm", int(s[[x for x in s.columns if x.startswith("x_")]].iloc[-1].sum()),
                            help=f"Tampon (m8i): {int(s['tampon'].iloc[-1])}")
                m[3].metric("Maliyet", f"{s['maliyet_birikimli'].iloc[-1]:,.0f} $".replace(",", "."))

    x0 = max(0.0, simdi - PENCERE_SAAT)
    adim_p = {k: a[a["t"] >= x0] for k, a in adim.items()}
    saat_p = {k: s[s["t"] >= x0] for k, s in saat.items() if not s.empty}
    x = eksen(x0, max(simdi, x0 + 1))
    bant = senaryo_bantlari(meta, bas, x0, max(simdi, x0 + 1), simdi)

    sol, sag = st.columns(2)
    with sol:
        st.markdown("**İşlemci: talep ve kapasite**")
        st.altair_chart(kapasite_grafigi(adim_p, "cpu", "Çekirdek", x, bant), use_container_width=True)
        st.markdown("**Yer bulamayan pod'lar (eksik kapasite)**")
        st.altair_chart(bekleyen_grafigi(adim_p, x, bant), use_container_width=True)
    with sag:
        st.markdown("**Bellek: talep ve kapasite**")
        st.altair_chart(kapasite_grafigi(adim_p, "mem", "GiB", x, bant), use_container_width=True)
        if saat_p:
            st.markdown("**Çalışan düğüm sayısı**")
            st.altair_chart(dugum_grafigi(saat_p, x, bant), use_container_width=True)


st.title("Canlı Kapasite Planlayıcı")
canli()
