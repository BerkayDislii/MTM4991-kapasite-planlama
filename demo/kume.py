"""Küme katmanı: planlayıcının "kaç düğüm, kaç pod" kararlarını uygular ve yer bulamayan pod'ları ölçer.

İki arka uç aynı arayüzü sunar:
  KubeKume   gerçek Kubernetes API'si (kind + KWOK). Düğümler KWOK'un yönettiği sahte Node nesneleridir;
             pod'ları gerçek kube-scheduler yerleştirir. Ölçüm, API'den okunan pod durumlarıdır.
  SahteKume  bellekte kapasite hesabı (kuru çalıştırma ve testler için; Kubernetes gerekmez).

Otel benzetmesi: düğüm = oda, pod = misafir, kube-scheduler = resepsiyon. Odası olmayan misafir lobide
bekler (Pending / Unschedulable); raporun "eksik kapasite"si budur.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from demo.cekirdek import Birim
from src.ilp.model import SunucuTipleri

ETIKET = "kapasite-demo"                 # bu demonun her nesnesindeki etiket anahtarı; değeri grup adıdır (ilp / r2)
UYGULAMALAR = ("islemci-yuk", "bellek-yuk")


@dataclass
class Olcum:
    """Bir adımdaki küme durumu."""
    bekleyen_cpu: int                    # yer bulamayan işlemci pod'u
    bekleyen_mem: int                    # yer bulamayan bellek pod'u
    bekleme_s: float                     # yerleşme için beklenen süre
    zaman_asimi: bool = False


def _kisa(tip_ad: str) -> str:
    return tip_ad.split(".")[0]          # c8i.xlarge → c8i


class SahteKume:
    """Kubernetes olmadan aynı hesabı yapan bellek içi küme (kuru çalıştırma)."""

    def __init__(self, tipler: SunucuTipleri, birim: Birim, **_):
        self.tipler, self.birim = tipler, birim
        self.pod_kap = birim.dugum_pod_kapasitesi(tipler)          # (K, 2)
        self.x = np.zeros(tipler.K, int)
        self.istek = (0, 0)

    def temizle(self) -> None:
        self.x[:] = 0
        self.istek = (0, 0)

    def dugumleri_ayarla(self, hedef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hedef = np.asarray(hedef, int)
        acilan, kapanan = np.maximum(hedef - self.x, 0), np.maximum(self.x - hedef, 0)
        self.x = hedef.copy()
        return acilan, kapanan

    def dugum_sayilari(self) -> np.ndarray:
        return self.x.copy()

    def yuku_ayarla(self, n_cpu: int, n_mem: int) -> None:
        self.istek = (n_cpu, n_mem)

    def yerlesmeyi_bekle(self, zaman_asimi_s: float = 0) -> Olcum:
        kap = (self.pod_kap * self.x[:, None]).sum(0)
        return Olcum(max(0, self.istek[0] - kap[0]), max(0, self.istek[1] - kap[1]), 0.0)


class KubeKume:
    """Gerçek Kubernetes kümesi (kind + KWOK)."""

    def __init__(self, tipler: SunucuTipleri, birim: Birim, grup: str, ad_alani: str, fiyat: np.ndarray):
        """grup: aynı kümede yan yana çalışan stratejileri ayırır (ör. "ilp", "r2"). Her grubun kendi
        düğümleri (etiket kapasite-demo=<grup>) ve kendi namespace'indeki yük Deployment'ları vardır;
        bir grubun pod'ları yalnız kendi grubunun düğümlerine yerleşebilir."""
        from kubernetes import client, config
        config.load_kube_config()
        self.c, self.core, self.apps = client, client.CoreV1Api(), client.AppsV1Api()
        self.tipler, self.birim, self.grup, self.ns, self.fiyat = tipler, birim, grup, ad_alani, fiyat
        self.pod_kap = birim.dugum_pod_kapasitesi(tipler)
        self.sayac = 0                                               # yeni düğüm adları için
        self.istek = (0, 0)
        self._kurulum()

    # ---------------------------------------------------------------- kurulum
    def _kurulum(self) -> None:
        """Namespace ve iki yük Deployment'ını (0 kopya) oluşturur; varsa dokunmaz."""
        c = self.c
        try:
            self.core.read_namespace(self.ns)
        except c.ApiException as h:
            if h.status != 404:
                raise
            self.core.create_namespace(c.V1Namespace(metadata=c.V1ObjectMeta(name=self.ns)))
        for ad, istek in zip(UYGULAMALAR, ({"cpu": _sayi(self.birim.cpu)},
                                            {"memory": f"{_sayi(self.birim.mem)}Gi"})):
            try:
                self.apps.read_namespaced_deployment(ad, self.ns)
                continue
            except c.ApiException as h:
                if h.status != 404:
                    raise
            sablon = c.V1PodTemplateSpec(
                metadata=c.V1ObjectMeta(labels={"app": ad, ETIKET: self.grup}),
                spec=c.V1PodSpec(
                    node_selector={ETIKET: self.grup},
                    tolerations=[c.V1Toleration(key="kwok.x-k8s.io/node", operator="Exists",
                                                effect="NoSchedule")],
                    termination_grace_period_seconds=0,
                    containers=[c.V1Container(name="is", image="sahte-imaj",
                                              resources=c.V1ResourceRequirements(requests=istek))]))
            dep = c.V1Deployment(
                metadata=c.V1ObjectMeta(name=ad, labels={ETIKET: self.grup}),
                spec=c.V1DeploymentSpec(replicas=0, selector=c.V1LabelSelector(match_labels={"app": ad}),
                                        template=sablon))
            self.apps.create_namespaced_deployment(self.ns, dep)

    # ---------------------------------------------------------------- düğümler
    def _dugumler(self) -> dict[int, list[str]]:
        """Tip dizini → bu demonun düğüm adları (eskiden yeniye)."""
        sonuc: dict[int, list[str]] = {k: [] for k in range(self.tipler.K)}
        for n in self.core.list_node(label_selector=f"{ETIKET}={self.grup}").items:
            k = self.tipler.ad.index(n.metadata.labels["node.kubernetes.io/instance-type"])
            sonuc[k].append(n.metadata.name)
        for k in sonuc:
            sonuc[k].sort()
        return sonuc

    def dugum_sayilari(self) -> np.ndarray:
        d = self._dugumler()
        return np.array([len(d[k]) for k in range(self.tipler.K)])

    def _dugum_ac(self, k: int) -> None:
        c, ad_tip = self.c, self.tipler.ad[k]
        self.sayac += 1
        ad = f"{self.grup}-{_kisa(ad_tip)}-{self.sayac:05d}"
        kap = {"cpu": _sayi(self.tipler.C[k]), "memory": f"{_sayi(self.tipler.M[k])}Gi", "pods": "110"}
        dugum = c.V1Node(
            metadata=c.V1ObjectMeta(
                name=ad,
                annotations={"kwok.x-k8s.io/node": "fake", "kapasite.ytu/fiyat-usd-saat": str(self.fiyat[k])},
                labels={ETIKET: self.grup, "node.kubernetes.io/instance-type": ad_tip,
                        "kubernetes.io/hostname": ad}),
            spec=c.V1NodeSpec(taints=[c.V1Taint(key="kwok.x-k8s.io/node", value="fake", effect="NoSchedule")]),
            status=c.V1NodeStatus(allocatable=kap, capacity=kap))
        self.core.create_node(dugum)

    def _dugum_kapat(self, ad: str) -> None:
        """Kubernetes'teki "drain" gibi: önce düğüm yeni pod almaz hâle getirilir (cordon), sonra üstündeki
        pod'lar silinir (Deployment yenilerini başka düğümlerde açar), en son düğüm silinir.

        Cordon olmadan arada bir yarış oluşur: silinen pod'un yerine açılan yeni pod, düğüm silinmeden hemen
        önce yine bu düğüme yerleşebilir ve var olmayan bir düğüme bağlı "yetim" pod kalır."""
        self.core.patch_node(ad, {"spec": {"unschedulable": True}})
        for pod in self.core.list_namespaced_pod(self.ns, field_selector=f"spec.nodeName={ad}").items:
            self.core.delete_namespaced_pod(pod.metadata.name, self.ns, grace_period_seconds=0)
        self.core.delete_node(ad)

    def dugumleri_ayarla(self, hedef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Her tipte düğüm sayısını hedefe getirir: önce açar, sonra (en yeni düğümden başlayarak) kapatır."""
        mevcut = self._dugumler()
        hedef = np.asarray(hedef, int)
        acilan, kapanan = np.zeros(self.tipler.K, int), np.zeros(self.tipler.K, int)
        for k in range(self.tipler.K):
            for _ in range(max(0, hedef[k] - len(mevcut[k]))):
                self._dugum_ac(k)
                acilan[k] += 1
        for k in range(self.tipler.K):
            fazla = len(mevcut[k]) - hedef[k]
            for ad in (mevcut[k][-fazla:] if fazla > 0 else []):
                self._dugum_kapat(ad)
                kapanan[k] += 1
        return acilan, kapanan

    # ---------------------------------------------------------------- yük
    def yuku_ayarla(self, n_cpu: int, n_mem: int) -> None:
        for ad, n in zip(UYGULAMALAR, (n_cpu, n_mem)):
            self.apps.patch_namespaced_deployment_scale(ad, self.ns, {"spec": {"replicas": int(n)}})
        self.istek = (int(n_cpu), int(n_mem))

    def _pod_durumu(self, dugumler: set[str] | None = None) -> dict[str, dict[str, int]]:
        """Uygulama başına: toplam (silinmekte olmayan), yerleşmiş, silinmekte olan pod sayısı.

        dugumler verilirse, var olmayan bir düğüme bağlı (yetim) pod yerleşmiş sayılmaz; silinir ki Deployment
        yerine yenisini açsın. Böylece ölçüm yalnızca gerçekten var olan kapasiteyi yansıtır."""
        yanit = self.core.list_namespaced_pod(self.ns, label_selector=f"{ETIKET}={self.grup}",
                                              _preload_content=False)
        durum = {a: {"toplam": 0, "yerlesik": 0, "siliniyor": 0} for a in UYGULAMALAR}
        for pod in json.loads(yanit.data)["items"]:
            d = durum.get(pod["metadata"]["labels"].get("app"))
            if d is None:
                continue
            if pod["metadata"].get("deletionTimestamp"):
                d["siliniyor"] += 1
                continue
            d["toplam"] += 1
            dugum = pod["spec"].get("nodeName")
            if dugum and dugumler is not None and dugum not in dugumler:
                self.core.delete_namespaced_pod(pod["metadata"]["name"], self.ns, grace_period_seconds=0)
                d["siliniyor"] += 1          # henüz durulmadı: yerine açılacak pod beklenir
                d["toplam"] -= 1
            elif dugum:
                d["yerlesik"] += 1
        return durum

    def yerlesmeyi_bekle(self, zaman_asimi_s: float = 30) -> Olcum:
        """Küme durulana kadar bekler, sonra yer bulamayan pod'ları sayar.

        "Duruldu" = istenen sayıda pod var, silinmekte olan pod kalmadı ve yerleşmiş pod sayısı artık
        değişmiyor (kapasite yetiyorsa hepsi, yetmiyorsa sığabilecek kadarı yerleşti). Beklenen değer yalnız
        ne zaman duracağımızı belirler; ölçülen sayı her zaman API'den okunan gerçek durumdur.
        """
        mevcut = self._dugumler()
        kap = (self.pod_kap * np.array([len(mevcut[k]) for k in range(self.tipler.K)])[:, None]).sum(0)
        adlar = set(sum(mevcut.values(), []))
        beklenen = {a: min(n, int(kp)) for a, n, kp in zip(UYGULAMALAR, self.istek, kap)}
        t0 = time.monotonic()
        while True:
            d = self._pod_durumu(adlar)
            tamam = all(d[a]["toplam"] == n and d[a]["siliniyor"] == 0 and d[a]["yerlesik"] == beklenen[a]
                        for a, n in zip(UYGULAMALAR, self.istek))
            gecen = time.monotonic() - t0
            if tamam or gecen > zaman_asimi_s:
                return Olcum(self.istek[0] - d["islemci-yuk"]["yerlesik"],
                             self.istek[1] - d["bellek-yuk"]["yerlesik"], gecen, zaman_asimi=not tamam)
            time.sleep(0.1)

    # ---------------------------------------------------------------- temizlik
    def temizle(self) -> None:
        """Bu demonun pod'larını ve düğümlerini kaldırır (önceki koşudan kalanlar dahil)."""
        self.yuku_ayarla(0, 0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:              # önce pod'lar gitsin (KWOK düğümleri varken siler)
            d = self._pod_durumu()
            if all(d[a]["toplam"] == 0 and d[a]["siliniyor"] == 0 for a in UYGULAMALAR):
                break
            time.sleep(0.2)
        for ad in sum(self._dugumler().values(), []):
            self.core.delete_node(ad)


def kumeyi_sifirla(onek: str, zaman_asimi_s: float = 120) -> None:
    """Önceki koşulardan kalan her şeyi kaldırır: adı <onek> ile başlayan namespace'lerdeki yük
    Deployment'ları (pod'larıyla), ardından bu demonun etiketini taşıyan tüm düğümler. Pod'lar düğümlerden önce
    silinir: KWOK pod'u ancak düğümü varken kapatabilir."""
    from kubernetes import client, config
    config.load_kube_config()
    core, apps = client.CoreV1Api(), client.AppsV1Api()
    for ns in core.list_namespace().items:
        if ns.metadata.name.startswith(onek):
            for dep in apps.list_namespaced_deployment(ns.metadata.name, label_selector=ETIKET).items:
                apps.delete_namespaced_deployment(dep.metadata.name, ns.metadata.name, propagation_policy="Foreground")
    t0 = time.monotonic()
    while (core.list_pod_for_all_namespaces(label_selector=ETIKET, limit=1).items
           or apps.list_deployment_for_all_namespaces(label_selector=ETIKET, limit=1).items) and \
            time.monotonic() - t0 < zaman_asimi_s:
        time.sleep(0.5)
    for n in core.list_node(label_selector=ETIKET).items:
        core.delete_node(n.metadata.name)


def _sayi(v: float) -> str:
    """4.0 → "4" (Kubernetes miktar biçimi)."""
    return str(int(v)) if float(v).is_integer() else str(v)
