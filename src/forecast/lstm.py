"""P2 LSTM (MIMARI.md Bölüm 5.2; kullanıcı talebi 2026-10-29: Bitbrains 48 sa, Alibaba 24 sa bakış penceresi).

Girdi : köken o öncesindeki lookback saatin 5 dk havuz toplamları (CPU, bellek kanalları), (L·12, 2)
Çıktı : sonraki 24 saatlik tepe (CPU + bellek), 48 değer. Kayıp MSE, AMP, erken durdurma (doğrulama).
Eğitim örnekleri saatlik kökenlerden üretilir; hedef penceresi tamamen ilgili bölümün içinde kalır.

Aşama 1: ölçekleyici yalnız eğitime uydurulur; doğrulama kaybıyla erken durdurma -> en iyi epoch sayısı.
         Eğitim / doğrulama ölçütleri bu modelle (doğrulama örneklem dışı).
Aşama 2: ölçekleyici eğitim+doğrulamaya uydurulur, model aynı tohumla en iyi epoch sayısı kadar eğitilir;
         test boyunca ağırlıklar sabit, yalnız girdi penceresi ilerler (Bölüm 5.1).

Çalıştırma: .venv/bin/python -m src.forecast.lstm
"""
import logging
import time
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn

from src.common.config import yaml_oku
from src.forecast.ortak import (Havuz, bolum_olcutleri, havuzlari_listele, modeller_dizini, saatlik_seri,
                                tahminleri_kaydet)
from src.forecast.veri import bes_dk_seri

log = logging.getLogger(__name__)

CIHAZ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Olcekleyici:
    """Kanal başına standartlaştırma; istatistikler yalnızca verilen (eğitim) diliminden."""

    def __init__(self, x: np.ndarray) -> None:
        self.ort = x.mean(axis=0)
        self.std = x.std(axis=0)
        assert (self.std > 0).all()

    def donustur(self, x: np.ndarray) -> np.ndarray:
        return (x - self.ort) / self.std

    def geri(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.ort


class LSTMTahminci(nn.Module):
    """Çok katmanlı LSTM -> son gizli durum -> doğrusal katman (ufuk × 2 çıktı)."""

    def __init__(self, gizli: int, katman: int, dropout: float, ufuk: int) -> None:
        super().__init__()
        self.ufuk = ufuk
        self.lstm = nn.LSTM(2, gizli, num_layers=katman, batch_first=True, dropout=dropout if katman > 1 else 0.0)
        self.cikis = nn.Linear(gizli, ufuk * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.cikis(hn[-1]).view(-1, self.ufuk, 2)


def ornekler(x5: np.ndarray, y: np.ndarray, kokenler: list[int], L: int, k: int,
             ufuk: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Köken o için girdi x5[(o-L)·k : o·k] ve hedef y[o : o+ufuk]. Girdi yalnız o öncesi veridir."""
    X = np.stack([x5[(o - L) * k: o * k] for o in kokenler])
    Y = np.stack([y[o: o + ufuk] for o in kokenler])
    return X.astype(np.float32), Y.astype(np.float32)


def tohumla(tohum: int) -> None:
    """Tüm rastgelelik kaynaklarını sabitler."""
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    torch.cuda.manual_seed_all(tohum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def egit(Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray | None, Yva: np.ndarray | None, cfg: dict[str, Any],
         tohum: int, epoch_sayisi: int | None = None) -> tuple[LSTMTahminci, dict[str, Any]]:
    """Modeli eğitir. Xva verilirse erken durdurma (en iyi ağırlıklar döner), yoksa epoch_sayisi kadar eğitir."""
    tohumla(tohum)
    model = LSTMTahminci(cfg["hidden_size"], cfg["num_layers"], cfg["dropout"], Ytr.shape[1]).to(CIHAZ)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    olcek = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    kayip_f = nn.MSELoss()
    Xt, Yt = torch.from_numpy(Xtr).to(CIHAZ), torch.from_numpy(Ytr).to(CIHAZ)
    gen = torch.Generator(device="cpu").manual_seed(tohum)
    n, bs = Xt.shape[0], cfg["batch_size"]
    max_ep = epoch_sayisi if epoch_sayisi is not None else cfg["max_epochs"]
    en_iyi = {"kayip": np.inf, "epoch": 0, "durum": None}
    son_tr = np.nan
    for ep in range(1, max_ep + 1):
        model.train()
        sira = torch.randperm(n, generator=gen).to(CIHAZ)
        toplam = 0.0
        for i in range(0, n, bs):
            idx = sira[i:i + bs]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=CIHAZ.type, dtype=torch.float16, enabled=cfg["amp"]):
                kayip = kayip_f(model(Xt[idx]).float(), Yt[idx])
            olcek.scale(kayip).backward()
            olcek.step(opt)
            olcek.update()
            toplam += float(kayip.detach()) * idx.numel()
        son_tr = toplam / n
        if Xva is None:
            continue
        va = float(np.mean((tahmin_et(model, Xva, cfg) - Yva) ** 2))
        if va < en_iyi["kayip"]:
            en_iyi = {"kayip": va, "epoch": ep, "durum": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif ep - en_iyi["epoch"] >= cfg["patience"] and ep >= cfg["min_epochs"]:
            break
    bilgi = {"son_egitim_kaybi": son_tr, "calisan_epoch": ep}
    if Xva is not None:
        model.load_state_dict(en_iyi["durum"])
        bilgi.update({"en_iyi_epoch": en_iyi["epoch"], "en_iyi_dogrulama_kaybi": en_iyi["kayip"]})
    return model, bilgi


@torch.no_grad()
def tahmin_et(model: LSTMTahminci, X: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Ölçekli uzayda tahmin: (n, ufuk, 2)."""
    model.eval()
    with torch.autocast(device_type=CIHAZ.type, dtype=torch.float16, enabled=cfg["amp"]):
        z = model(torch.from_numpy(X).to(CIHAZ)).float()
    return z.cpu().numpy()


def havuz_calistir(h: Havuz, cfg: dict[str, Any], k: int) -> tuple[list[dict], list[dict]]:
    """Bir havuzda tüm tohumlar için iki aşamalı eğitim; ölçüt ve eğitim günlüğü satırları döner."""
    y, x5 = saatlik_seri(h), bes_dk_seri(h)
    L = h.lookback
    t_son, v_son = h.sinir["train"][1], h.sinir["val"][1]
    gunler = h.degerlendirme_gunleri
    kok = {b: [24 * g for g in gunler[b]] for b in gunler}
    olcut, gunluk = [], []
    for tohum in cfg["seeds"]:
        t0 = time.time()
        # ---- Aşama 1: eğitim ölçekleyicisi, erken durdurma
        sc = Olcekleyici(x5[:t_son * k])
        z5, zy = sc.donustur(x5), sc.donustur(y)
        Xtr, Ytr = ornekler(z5, zy, list(range(L, t_son - 24 + 1)), L, k)
        Xva, Yva = ornekler(z5, zy, list(range(t_son, v_son - 24 + 1)), L, k)
        model, bilgi = egit(Xtr, Ytr, Xva, Yva, cfg, tohum)
        tahmin = {b: sc.geri(tahmin_et(model, ornekler(z5, zy, kok[b], L, k)[0], cfg)) for b in ("train", "val")}
        # ---- Aşama 2: eğitim+doğrulama ölçekleyicisi, sabit epoch sayısı
        sc2 = Olcekleyici(x5[:v_son * k])
        z5, zy = sc2.donustur(x5), sc2.donustur(y)
        Xtv, Ytv = ornekler(z5, zy, list(range(L, v_son - 24 + 1)), L, k)
        model2, bilgi2 = egit(Xtv, Ytv, None, None, cfg, tohum, epoch_sayisi=bilgi["en_iyi_epoch"])
        tahmin["test"] = sc2.geri(tahmin_et(model2, ornekler(z5, zy, kok["test"], L, k)[0], cfg))

        ad = f"lstm_s{tohum}"
        tahminleri_kaydet(h, ad, tahmin)
        olcut += bolum_olcutleri(h, y, tahmin, "lstm", tohum)
        gunluk.append({"havuz": h.ad, "tohum": tohum, "lookback_saat": L, "n_egitim_ornek": len(Xtr),
                       "n_dogrulama_ornek": len(Xva), "n_son_egitim_ornek": len(Xtv), **bilgi,
                       "son_model_egitim_kaybi": bilgi2["son_egitim_kaybi"], "sure_s": round(time.time() - t0, 1)})
        log.info("%s tohum %d: en iyi epoch %d, doğrulama MSE %.4f, %.1f s", h.ad, tohum, bilgi["en_iyi_epoch"],
                 bilgi["en_iyi_dogrulama_kaybi"], time.time() - t0)
    return olcut, gunluk


def main() -> None:
    """Tüm havuzlarda LSTM; outputs/models/metrikler_lstm.csv ve lstm_egitim.csv yazar."""
    cfg = yaml_oku("forecast")["lstm"]
    assert cfg["input_resolution_steps"] == 1, "Girdi çözünürlüğünü düşürmek MIMARI.md'ye göre sapmadır"
    k = yaml_oku("preprocess")["grid"]["steps_per_slot"]
    log.info("Cihaz: %s", torch.cuda.get_device_name(0) if CIHAZ.type == "cuda" else "cpu")
    olcut, gunluk = [], []
    for h in havuzlari_listele():
        o, g = havuz_calistir(h, cfg, k)
        olcut += o
        gunluk += g
    md = modeller_dizini()
    pl.DataFrame(olcut, infer_schema_length=None).write_csv(md / "metrikler_lstm.csv")
    pl.DataFrame(gunluk, infer_schema_length=None).write_csv(md / "lstm_egitim.csv")
    log.info("LSTM ölçüt ve eğitim günlüğü yazıldı; tepe VRAM %.0f MiB", torch.cuda.max_memory_allocated() / 2**20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
