"""M6 tepkisel/hibrit strateji testleri (MIMARI.md Bölüm 7)."""
import numpy as np
import pytest

from src.baselines.tepkisel import EsikParam, esik_simule, tepkisel_ilp
from src.evaluate import olay_olcutleri, plan_olcutleri
from src.ilp.model import IlpParam, Plan, SunucuTipleri, dogrula


def tek_tip(C: float = 10, M: float = 10, c: float = 1.0) -> SunucuTipleri:
    return SunucuTipleri(("a",), np.array([C], float), np.array([M], float), np.array([c], float))


def param(t: SunucuTipleri, alpha: float = 0.0, w: float = 0.0, L: int = 1) -> IlpParam:
    return IlpParam(alpha=alpha, s=t.c * w, d=t.c * 0.0, L=np.full(t.K, L))


ESIK = EsikParam(tip=0, ust=0.8, hedef=0.7, alt=0.5, soguma=2)


def sifir_plan(K: int, T: int) -> Plan:
    return Plan(*(np.zeros((K, T)) for _ in range(3)), amac=0.0)


def talep(cpu: list[float]) -> np.ndarray:
    """CPU serisi; bellek 0 (yalnız CPU bağlayıcı)."""
    return np.column_stack([np.array(cpu, float), np.zeros(len(cpu))])


def test_olay_olcutleri() -> None:
    """Art arda eksik dilimler tek olaydır; süreler dilim cinsinden."""
    o = olay_olcutleri(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], bool))
    assert o == {"n_olay": 3.0, "olay_sure_ort": 2.0, "olay_sure_max": 3.0}
    assert olay_olcutleri(np.zeros(4, bool)) == {"n_olay": 0.0, "olay_sure_ort": 0.0, "olay_sure_max": 0.0}
    assert olay_olcutleri(np.ones(3, bool))["olay_sure_max"] == 3.0


def test_r1_basamakta_bir_dilim_gecikme() -> None:
    """Basamak artışında eşik kuralı bir dilim geç kalır, sonraki dilimde hedef kullanıma göre büyür."""
    t, p = tek_tip(), param(tek_tip())
    D = talep([14, 14, 14, 30, 30, 30])
    plan = esik_simule(D, D[0], sifir_plan(1, 6), t, p, ESIK, min_tampon=1)
    # ceil(14 / (0,7·10)) = 2; basamaktan sonra ceil(30 / 7) = 5
    assert plan.x[0].tolist() == [2, 2, 2, 2, 5, 5]
    o = plan_olcutleri(plan, D[:, 0], D[:, 1], t, p)
    assert o["eksik_saat"] == 1 and o["n_olay"] == 1 and o["olay_sure_max"] == 1


def test_r1_kucultme_soguma_ve_min_calisma() -> None:
    """Düşüşte küçültme soğuma dolmadan ve L dolmadan yapılmaz."""
    t = tek_tip()
    p = param(t, L=3)
    D = talep([30, 30, 2, 2, 2, 2])
    plan = esik_simule(D, np.array([0.0, 0.0]), sifir_plan(1, 6), t, p, ESIK, min_tampon=1)
    # t=0: soğuk başlangıç, D_once = 0 → yalnız min 1 (açılış t=0); t=1: u = 30/10 > 0,8 → 5 (4 açılış t=1);
    # t=2: u = 30/50 = 0,6 → değişmez; t=3: u = 0,04 < 0,5, soğuma doldu, hedef 1 ama L = 3 t=1'deki 4 açılışı
    # tutar → 4 (yalnız t=0 sunucusu kapanır); t=4: son değişim t=3, soğuma dolmadı → 4; t=5: → 1
    assert plan.x[0].tolist() == [1, 5, 5, 4, 4, 1]
    assert not dogrula(plan, np.zeros(6), np.zeros(6), t, p)


def test_r2_plan_tabaninin_altina_inmez() -> None:
    """Hibritte toplam sunucu her dilimde ILP planı kadar ya da fazladır; plan yeterliyse tampon 0'dır."""
    t, p = tek_tip(), param(tek_tip())
    baz = Plan(np.array([[5, 5, 5, 5, 5]], float), np.array([[5, 0, 0, 0, 0]], float), np.zeros((1, 5)), 0.0)
    D = talep([10, 10, 60, 60, 10])
    plan = esik_simule(D, D[0], baz, t, p, ESIK)
    assert (plan.x >= baz.x).all()
    # t=3: u = 60/50 > 0,8 → ceil((60/0,7 − 50)/10) = 4 tampon; t=4: u = 60/90 → değişmez
    assert plan.x[0].tolist() == [5, 5, 5, 9, 9]
    # Sıfır taban ve min 0 tampon: plan yokken R1 ile aynı
    r1 = esik_simule(D, D[0], sifir_plan(1, 5), t, p, ESIK, min_tampon=0)
    assert (esik_simule(D, D[0], sifir_plan(1, 5), t, p, ESIK).x == r1.x).all()


def test_r0_yalniz_gecmis_talebi_gorur() -> None:
    """Tepkisel-ILP'de t dilimi kararı D_t'den bağımsızdır (sızıntı yok): D_t değişince x_t değişmez."""
    t = tek_tip()
    p = param(t, alpha=0.05, w=0.25, L=2)
    D = talep([10, 20, 35, 5, 5])
    a, girdi = tepkisel_ilp(D, np.array([8.0, 0.0]), t, p, "SCIP", 10)
    assert girdi[:, 0].tolist() == [8, 10, 20, 35, 5]
    D2 = D.copy()
    D2[-1, 0] = 500
    b, _ = tepkisel_ilp(D2, np.array([8.0, 0.0]), t, p, "SCIP", 10)
    assert (a.x == b.x).all()
    # ceil(1,05·girdi / 10); t=4'te girdi 5 → 1 sunucu yeter, L = 2 yalnız t=3'te açılan 1 sunucuyu tutar
    assert a.x[0].tolist() == [1, 2, 3, 4, 1]
    assert not dogrula(a, girdi[:, 0], girdi[:, 1], t, p)
    assert plan_olcutleri(a, D[:, 0], D[:, 1], t, p)["eksik_saat"] == pytest.approx(1)


def test_r2_uyumlu_esik_hedefi_alfaya_bagli() -> None:
    """hibrit_uyumlu hedefi null iken 1/(1+α) olur ve eşik sırası korunur."""
    from src.baselines.tepkisel import esik_param_oku
    from src.common.config import yaml_oku
    from src.ilp.model import tipleri_oku
    e = esik_param_oku(tipleri_oku(), uyumlu=True)
    if yaml_oku("ilp")["tepkisel"]["hibrit_uyumlu"]["esik_hedef"] is None:
        assert e.hedef == pytest.approx(1 / (1 + yaml_oku("ilp")["model"]["alpha"]))
    assert e.alt < e.hedef <= e.ust <= 1


def test_pareto_isaretle() -> None:
    """Pareto: ikisinde birden kötü ya da eşit (birinde kesin kötü) olan nokta elenir."""
    import polars as pl
    from src.izgara import pareto_isaretle
    df = pl.DataFrame({"grup": ["g"] * 4, "strateji": ["s"] * 4, "model": ["m"] * 4,
                       "maliyet_gunluk_ort": [1.0, 2.0, 3.0, 2.0], "eksik_saat_orani_ort": [0.5, 0.2, 0.0, 0.3]})
    assert pareto_isaretle(df)["pareto"].to_list() == [True, True, True, False]
