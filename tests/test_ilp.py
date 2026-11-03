"""ILP modeli ve değerlendirme testleri (MIMARI.md Bölüm 11, ILP (a)–(g) + değerlendirme)."""
import numpy as np
import pytest

from src.evaluate import plan_olcutleri
from src.ilp.model import (IlpParam, Plan, SunucuTipleri, amac_hesapla, coz, coz_cpsat, devralinan, dogrula,
                           lp_yuvarla, tipleri_oku)


def tek_tip(C: float = 10, M: float = 10, c: float = 1.0) -> SunucuTipleri:
    return SunucuTipleri(("a",), np.array([C], float), np.array([M], float), np.array([c], float))


def param(t: SunucuTipleri, alpha: float = 0.0, w: float = 0.0, L: int = 1, w_kapatma: float = 0.0) -> IlpParam:
    return IlpParam(alpha=alpha, s=t.c * w, d=t.c * w_kapatma, L=np.full(t.K, L))


def test_a_bilinen_optimum() -> None:
    """(a) Elle çözülmüş örnek: CPU 8, bellek 40 → 1 A (4/8) + 1 B (4/32) = 2,5 en ucuz."""
    t = SunucuTipleri(("A", "B"), np.array([4.0, 4.0]), np.array([8.0, 32.0]), np.array([1.0, 1.5]))
    plan = coz(np.array([8.0]), np.array([40.0]), t, param(t))
    assert plan.x[:, 0].tolist() == [1, 1] and plan.amac == pytest.approx(2.5)
    # Tek tip, α = 0,05: 10·1,05 = 10,5 çekirdek → ceil(10,5/4) = 3 sunucu
    t1 = tek_tip(C=4, M=100)
    assert coz(np.array([10.0]), np.array([0.0]), t1, param(t1, alpha=0.05)).x[0, 0] == 3


def test_b_minimum_calisma_suresi() -> None:
    """(b) Tek dilimlik sıçrama: açılan sunucu L = 4 dilim açık kalır."""
    t = tek_tip()
    # Sıçrama ilk dilimde: açılış zamanı tek (sonraki dilimde sıçrama, t=0 açılışıyla eşdeğer optimum verirdi)
    D = np.array([10, 0, 0, 0, 0, 0, 0], float)
    plan = coz(D, np.zeros_like(D), t, param(t, w=0.5, L=4))
    assert plan.x[0].tolist() == [1, 1, 1, 1, 0, 0, 0]
    # L = 1 iken hemen kapanır
    assert coz(D, np.zeros_like(D), t, param(t, w=0.5, L=1)).x[0].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert not dogrula(plan, D, np.zeros_like(D), t, param(t, w=0.5, L=4))


def test_b_devralinan_baslatma() -> None:
    """(b) Önceki günün son diliminde açılan sunucu (L = 4) yeni günün ilk 3 dilimi boyunca kapatılamaz."""
    t = tek_tip()
    p = param(t, w=0.5, L=4)
    y_onceki = np.zeros((1, 24))
    y_onceki[0, -1] = 2
    assert devralinan(y_onceki, p.L, 24)[0, :4].tolist() == [2, 2, 2, 0]
    D = np.zeros(6)
    plan = coz(D, D, t, p, x0=np.array([2.0]), y_onceki=y_onceki)
    assert plan.x[0].tolist() == [2, 2, 2, 0, 0, 0]
    # Doğrulayıcı devralınanı hesaba katmadan erken kapatmayı yakalar
    erken = Plan(np.zeros((1, 6)), np.zeros((1, 6)), np.array([[2, 0, 0, 0, 0, 0]], float), 0.0)
    assert any(i.startswith("(5)") for i in dogrula(erken, D, D, t, p, np.array([2.0]), y_onceki))


def test_c_yuksek_acma_maliyeti_dalgalanmayi_onler() -> None:
    """(c) Dalgalı talepte s_k = 0 ise sunucu sayısı dalgalanır; s_k büyükse sabit kalır."""
    t = tek_tip()
    D = np.array([10, 0] * 6, float)
    ucuz = coz(D, np.zeros_like(D), t, param(t, w=0.0))
    pahali = coz(D, np.zeros_like(D), t, param(t, w=10.0))
    assert ucuz.y.sum() == 6 and pahali.y.sum() == 1
    assert (pahali.x[0, :-1] == 1).all()


def test_d_statik_ozel_durum() -> None:
    """(d) s = d = 0, L = 1 → çok dönemli maliyet = dönem dönem bağımsız çözümlerin toplamı."""
    t = tipleri_oku()
    rng = np.random.default_rng(0)
    Dc, Dm = rng.uniform(20, 200, 12), rng.uniform(50, 1500, 12)
    p = param(t, alpha=0.05)
    cok = coz(Dc, Dm, t, p)
    tek = sum(coz(Dc[i:i + 1], Dm[i:i + 1], t, p).amac for i in range(12))
    assert cok.amac == pytest.approx(tek, rel=1e-9)
    # Bölüm 6.3: s = d = 0 iken raporlanan açma sayısı max(0, Δx)
    fark = np.diff(np.concatenate([np.zeros((t.K, 1)), cok.x], axis=1), axis=1)
    assert np.array_equal(cok.y, np.maximum(fark, 0)) and np.array_equal(cok.z, np.maximum(-fark, 0))


def test_e_dogrulayici() -> None:
    """(e) Doğrulayıcı (2)–(6) ihlallerini bağımsız yakalar; geçerli planda boş liste döner."""
    t = SunucuTipleri(("A", "B"), np.array([4.0, 4.0]), np.array([8.0, 32.0]), np.array([1.0, 1.5]))
    p = param(t, w=0.5, L=2)
    Dc, Dm = np.array([8.0, 8.0, 0.0]), np.array([10.0, 40.0, 0.0])
    plan = coz(Dc, Dm, t, p)
    assert dogrula(plan, Dc, Dm, t, p) == []
    assert amac_hesapla(plan, t, p) == pytest.approx(plan.amac)

    def boz(**degis) -> list[str]:
        q = Plan(plan.x.copy(), plan.y.copy(), plan.z.copy(), plan.amac)
        for v, (k, i, deger) in degis.items():
            getattr(q, v)[k, i] = deger
        return dogrula(q, Dc, Dm, t, p)

    assert any(i.startswith("(2)") for i in boz(x=(0, 0, 0), y=(0, 0, 0)))
    assert any(i.startswith("(3)") for i in boz(x=(1, 1, 0)))
    assert any(i.startswith("(4)") for i in boz(y=(0, 2, 5)))
    assert any(i.startswith("(6)") for i in boz(x=(0, 2, 0.5)))
    # (5): t=1'de açılan sunucuyu t=2'de kapat (L = 2 ihlali)
    q = Plan(np.array([[0, 1, 0]], float), np.array([[0, 1, 0]], float), np.array([[0, 0, 1]], float), 0.0)
    t1 = tek_tip()
    assert any(i.startswith("(5)") for i in dogrula(q, np.zeros(3), np.zeros(3), t1, param(t1, L=2)))


def test_f_scip_cpsat_ayni_optimum() -> None:
    """(f) Gerçek sunucu tipleriyle küçük örnekte SCIP ve CP-SAT aynı optimum maliyeti verir."""
    t = tipleri_oku()
    rng = np.random.default_rng(1)
    for alpha, w, L in ((0.05, 0.25, 2), (0.0, 1.0, 4), (0.1, 0.0, 1)):
        Dc, Dm = rng.uniform(10, 120, 8), rng.uniform(20, 700, 8)
        p = param(t, alpha=alpha, w=w, L=L)
        scip, cps = coz(Dc, Dm, t, p), coz_cpsat(Dc, Dm, t, p)
        assert scip.amac == pytest.approx(cps.amac, rel=1e-9, abs=1e-9)
        assert dogrula(cps, Dc, Dm, t, p) == []


def test_g_lp_yuvarlama_farki() -> None:
    """(g) LP + yukarı yuvarlama: maliyet ILP'den asla düşük değildir ve bu örnekte kesin olarak yüksektir."""
    t = tipleri_oku()
    p = param(t, alpha=0.05, w=0.25, L=2)
    Dc, Dm = np.array([10.0, 30.0, 18.0]), np.array([50.0, 60.0, 120.0])
    ilp = coz(Dc, Dm, t, p)
    lp, yuv, uygun = lp_yuvarla(Dc, Dm, t, p)
    assert lp.amac <= ilp.amac + 1e-9
    assert yuv.amac >= ilp.amac - 1e-9
    assert yuv.amac > ilp.amac + 1e-6, "Bu örnekte yuvarlama optimum olmamalı"
    print(f"ILP={ilp.amac:.5f} LP={lp.amac:.5f} LP+yuvarlama={yuv.amac:.5f} "
          f"(+%{100 * (yuv.amac - ilp.amac) / ilp.amac:.2f}, (5) uygun={uygun})")


def test_degerlendirme_gercek_talep_alfasiz() -> None:
    """Yetersizlik sayacı gerçek D ve α'sız kapasiteyle çalışır; boşta oran yetersiz dilimde 0'dır."""
    t = tek_tip(C=10, M=20, c=2.0)
    p = param(t, alpha=0.5, w=0.5)
    plan = Plan(np.array([[1, 2, 1]], float), np.array([[1, 1, 0]], float), np.array([[0, 0, 1]], float), 0.0)
    o = plan_olcutleri(plan, np.array([12.0, 15.0, 10.0]), np.array([5.0, 10.0, 25.0]), t, p)
    assert o["maliyet"] == pytest.approx(2 * 4 + 1.0 * 2)
    assert o["eksik_saat_cpu"] == 1 and o["karsilanmayan_cpu"] == pytest.approx(2.0)
    assert o["eksik_saat_mem"] == 1 and o["karsilanmayan_mem"] == pytest.approx(5.0)
    assert o["eksik_saat"] == 2 and o["eksik_saat_orani"] == pytest.approx(2 / 3)
    assert o["bos_orani_cpu"] == pytest.approx((0 + 5 / 20 + 0) / 3)
    assert o["bos_orani_mem"] == pytest.approx((15 / 20 + 30 / 40 + 0) / 3)
