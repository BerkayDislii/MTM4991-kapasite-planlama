import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.forecast.ortak import havuzlari_listele, saatlik_seri
from src.forecast.saatlik import sarima_kokenler
from src.ilp.model import tipleri_oku, param_oku, coz, dogrula
h = [x for x in havuzlari_listele() if x.ad == "bitbrains_fastStorage"][0]
y = saatlik_seri(h)
O = h.sinir["val"][0]                      # doğrulama döneminin ilk günü
T = sarima_kokenler(h, y, [O])[0]          # (24, 2)
tip = tipleri_oku(); p = param_oku(tip)
plan = coz(T[:, 0], T[:, 1], tip, p)
assert not dogrula(plan, T[:, 0], T[:, 1], tip, p)
print("sure_ms", plan.bilgi, "amac", plan.amac, "x tipler", plan.x.sum(1), "acilis", plan.y.sum(1))
kap_c = (tip.C[:, None] * plan.x).sum(0); kap_m = (tip.M[:, None] * plan.x).sum(0)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 11})
fig, ax = plt.subplots(3, 1, figsize=(8, 8.2), sharex=True)
t = np.arange(24); renk = ["#4a89dc", "#f08050", "#40b88a"]
alt = np.zeros(24)
for k in range(tip.K):
    ax[0].bar(t, plan.x[k], bottom=alt, width=0.9, color=renk[k], label=tip.ad[k]); alt += plan.x[k]
ax[0].set_ylabel("Sunucu sayısı"); ax[0].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
for a, kap, D, birim in ((ax[1], kap_c, T[:, 0], "İşlemci (çekirdek)"), (ax[2], kap_m, T[:, 1], "Bellek (GB)")):
    a.step(t, kap, where="mid", color="#1f6fd1", lw=1.8, label="ILP kapasitesi")
    a.step(t, (1 + p.alpha) * D, where="mid", color="k", lw=1.2, ls="--", label="(1 + α) × SARIMA tahmini")
    a.set_ylabel(birim); a.grid(alpha=.3)
ax[1].legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
ax[2].set_xlabel("Günün saati"); ax[2].set_xticks(range(0, 24, 2))
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("/home/claude/ara1/sekiller/ornek_gunluk_plan.png", dpi=220)
print(fig.get_size_inches()*220)
