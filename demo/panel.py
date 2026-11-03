"""Canlı panel: demonun yazdığı CSV'leri 2 saniyede bir okuyup çizer.

Çalıştırma (proje kökünden, demo çalışırken ayrı bir terminalde):
    .venv/bin/streamlit run demo/panel.py
Tarayıcıda http://localhost:8501 açılır.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

KOK = Path(__file__).resolve().parents[1]
with open(Path(__file__).with_name("ayar.yaml"), encoding="utf-8") as f:
    CIKTI = KOK / yaml.safe_load(f)["cikti"]
PENCERE_SAAT = 48                                   # grafiklerde gösterilen son saat sayısı
RENK = {"Talep": "#2A6FC4", "Kapasite": "#C2501F", "Plan (ILP)": "#8A94A6", "c8i": "#2A6FC4", "m8i": "#5FA36A", "r8i": "#8E6BB8",
        "İşlemci pod'u": "#2A6FC4", "Bellek pod'u": "#C2501F"}
STRATEJI_AD = {"ilp": "Yalnız ILP planı", "r2_uyumlu": "ILP planı + uyumlu tampon (R2)"}

st.set_page_config(page_title="Kapasite Planlayıcı", layout="wide")
st.markdown("<style>div.block-container{padding-top:1.5rem} [data-testid='stMetricValue']{font-size:1.9rem}"
            "</style>", unsafe_allow_html=True)


def oku() -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
    try:
        meta = json.loads((CIKTI / "durum.json").read_text(encoding="utf-8"))
        adim = pd.read_csv(CIKTI / "adim.csv")
        saat = pd.read_csv(CIKTI / "saat.csv")
    except (FileNotFoundError, json.JSONDecodeError, pd.errors.EmptyDataError):
        return None, pd.DataFrame(), pd.DataFrame()
    return meta, adim, saat


def renk(alanlar: list[str]) -> alt.Scale:
    return alt.Scale(domain=alanlar, range=[RENK[a] for a in alanlar])


def x_ekseni(alan: pd.DataFrame) -> alt.X:
    return alt.X("t:Q", title="Benzetim saati", scale=alt.Scale(domain=[alan["t"].min(), alan["t"].max() + 1 / 12],
                                                                nice=False))


def talep_kapasite(adim: pd.DataFrame, kaynak: str, birim: str) -> alt.Chart:
    """Talep (5 dk), kümedeki kapasite ve yalnız ILP planının kapasitesi (aradaki fark = tampon)."""
    seriler = ["Talep", "Kapasite", "Plan (ILP)"]
    d = adim[["t", f"talep_{kaynak}", f"kap_{kaynak}", f"plan_{kaynak}"]].rename(
        columns={f"talep_{kaynak}": "Talep", f"kap_{kaynak}": "Kapasite", f"plan_{kaynak}": "Plan (ILP)"}).melt(
        "t", var_name="Seri", value_name="v")
    return alt.Chart(d).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=x_ekseni(adim), y=alt.Y("v:Q", title=birim),
        color=alt.Color("Seri:N", scale=renk(seriler), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Seri:N", scale=alt.Scale(domain=seriler, range=[[1, 0], [1, 0], [5, 4]]),
                                  legend=None),
    ).properties(height=260)


def dugumler(saat: pd.DataFrame, kisa: list[str]) -> alt.Chart:
    son = saat.iloc[[-1]].assign(t=saat["t"].iloc[-1] + 1)     # son saati de tam genişlikte çiz
    s = pd.concat([saat, son])
    d = s[["t"] + [f"x_{k}" for k in kisa]].rename(columns={f"x_{k}": k for k in kisa}).melt(
        "t", var_name="Tip", value_name="Düğüm")
    return alt.Chart(d).mark_area(interpolate="step-after").encode(
        x=x_ekseni(s), y=alt.Y("Düğüm:Q", stack=True, title="Çalışan düğüm"),
        color=alt.Color("Tip:N", scale=renk(kisa), legend=alt.Legend(orient="top", title=None)),
    ).properties(height=260)


def bekleyen(adim: pd.DataFrame) -> alt.Chart:
    d = adim[["t", "bekleyen_cpu", "bekleyen_mem"]].rename(
        columns={"bekleyen_cpu": "İşlemci pod'u", "bekleyen_mem": "Bellek pod'u"}).melt(
        "t", var_name="Tür", value_name="Pod")
    return alt.Chart(d).mark_bar(width=3).encode(
        x=x_ekseni(adim), y=alt.Y("Pod:Q", stack=True, title="Yer bulamayan pod"),
        color=alt.Color("Tür:N", scale=renk(["İşlemci pod'u", "Bellek pod'u"]),
                        legend=alt.Legend(orient="top", title=None)),
    ).properties(height=260)


@st.cache_data
def benzetim(havuz: str, strateji: str, model: str) -> pd.DataFrame | None:
    """Raporun benzetimi (outputs/reactive/sim/...): saat başına eksik mi ve maliyet. Demo ile aynı saatleri
    karşılaştırmak için kullanılır."""
    ad = {"ilp": "ilp", "r2_uyumlu": "r2_hibrit_uyumlu"}[strateji]
    try:
        sim = pd.read_parquet(KOK / "outputs" / "reactive" / "sim" / "bitbrains" / havuz / f"{ad}_{model}.parquet")
        with open(KOK / "configs" / "ilp.yaml", encoding="utf-8") as f:
            w = yaml.safe_load(f)["model"]["w"]
        with open(KOK / "configs" / "server_types.yaml", encoding="utf-8") as f:
            fiyat = {t["ad"]: t["saatlik_usd"] for t in yaml.safe_load(f)["tipler"]}
    except (FileNotFoundError, KeyError):
        return None
    sim["eksik"] = (sim["cpu_peak"] > sim["kap_cpu"] + 1e-9) | (sim["mem_peak"] > sim["kap_mem"] + 1e-9)
    sim["maliyet"] = sum(c * (sim[f"x_{a}"] + w * sim[f"y_{a}"]) for a, c in fiyat.items())
    return sim[["slot", "eksik", "maliyet"]]


@st.fragment(run_every="2s")
def canli() -> None:
    meta, adim, saat = oku()
    if meta is None or adim.empty:
        st.info("Demo henüz başlamadı. Başka bir terminalde: `.venv/bin/python -m demo.calistir`")
        return
    kisa = [t.split(".")[0] for t in meta["tipler"]]
    bas = meta["bas"]
    adim["t"] = adim["slot"] - bas + adim["adim"] / 12
    saat["t"] = saat["slot"] - bas
    C = dict(zip(kisa, meta.get("cekirdek", [4] * len(kisa))))
    M = dict(zip(kisa, meta.get("bellek_gib", [8, 16, 32])))
    saat["plan_cpu"] = sum(saat[f"plan_{t}"] * C[t] for t in kisa)
    saat["plan_mem"] = sum(saat[f"plan_{t}"] * M[t] for t in kisa)
    adim = adim.merge(saat[["slot", "plan_cpu", "plan_mem"]], on="slot", how="left")
    for r in ("cpu", "mem"):                         # saati henüz bitmemiş adımlar: plan = kapasite (geçici)
        adim[f"plan_{r}"] = adim[f"plan_{r}"].fillna(adim[f"kap_{r}"])

    st.caption(f"{meta['havuz']} · tahmin: {meta['model'].upper()} · {STRATEJI_AD[meta['strateji']]} · "
               f"{'tamamlandı' if meta['bitti'] else meta['mesaj']}")

    son = adim.iloc[-1]
    k = st.columns(3) + st.columns(3)
    k[0].metric("Benzetim zamanı", f"Gün {int(son['gun'])} · {int(son['saat']):02d}:{int(son['adim']) * 5:02d}")
    if not saat.empty:
        s = saat.iloc[-1]
        k[1].metric("Çalışan düğüm", int(sum(s[f"x_{t}"] for t in kisa)),
                    help=" · ".join(f"{t}: {int(s[f'x_{t}'])}" for t in kisa))
        k[2].metric("Tampon (m8i)", int(s["tampon"]))
        n_eksik, n = int(saat["eksik"].sum()), len(saat)
        sim = benzetim(meta["havuz"], meta["strateji"], meta["model"])
        sim = None if sim is None else sim[sim["slot"].isin(saat["slot"])]
        k[4].metric("Eksik kapasiteli saat", f"{n_eksik} / {n}")
        k[4].caption(f"%{100 * n_eksik / n:.1f}".replace(".", ",")
                     + (f" · raporun benzetimi, aynı saatler: {int(sim['eksik'].sum())} / {len(sim)}"
                        if sim is not None else ""))
        k[5].metric("Maliyet (birikimli)", f"{s['maliyet_birikimli']:,.0f} USD".replace(",", "."))
        k[5].caption(f"{s['maliyet_birikimli'] / n * 24:,.0f} USD/gün".replace(",", ".")
                     + (f" · benzetim: {sim['maliyet'].sum():,.0f} USD".replace(",", ".") if sim is not None else ""))
    k[3].metric("Yer bulamayan pod", int(son["bekleyen_cpu"] + son["bekleyen_mem"]))

    ust = adim["t"].max()
    a = adim[adim["t"] > ust - PENCERE_SAAT]
    s = saat[saat["t"] > ust - PENCERE_SAAT]
    sol, sag = st.columns(2)
    with sol:
        st.markdown("**İşlemci: talep ve kapasite**")
        st.altair_chart(talep_kapasite(a, "cpu", "Çekirdek"), use_container_width=True)
        st.markdown("**Çalışan düğümler (tipe göre)**")
        st.altair_chart(dugumler(s, kisa), use_container_width=True)
    with sag:
        st.markdown("**Bellek: talep ve kapasite**")
        st.altair_chart(talep_kapasite(a, "mem", "GiB"), use_container_width=True)
        st.markdown("**Yer bulamayan pod'lar (eksik kapasite)**")
        st.altair_chart(bekleyen(a), use_container_width=True)


st.title("Canlı Kapasite Planlayıcı")
canli()
