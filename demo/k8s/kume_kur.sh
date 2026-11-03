#!/usr/bin/env bash
# Demo kümesini hazırlar: kind kümesi + KWOK + kontrol düzleminin hız ayarı.
# Tekrar çalıştırmak güvenlidir: var olan küme silinmez, yapılmış ayarlar yeniden yapılmaz.
#
# Hız ayarı neden gerekli? Varsayılan ayarlarla controller-manager API'ye saniyede ~20 istek gönderir.
# Bir 5 dakikalık adımda yüz pod açılıp kapandığında bu sınır adımı saniyelerce geciktirir. Sınırı yükseltmek
# yalnız hızı değiştirir, kararları değil (gerçek kümelerde de büyük ölçekte bu ayar yapılır).
set -euo pipefail

KUME="${KUME:-kapasite}"
DUGUM="${KUME}-control-plane"
KWOK_URL="https://github.com/kubernetes-sigs/kwok/releases/latest/download"

if kind get clusters 2>/dev/null | grep -qx "$KUME"; then
  echo "• '$KUME' kümesi zaten var"
else
  echo "• '$KUME' kümesi kuruluyor (1–2 dk)"
  kind create cluster --name "$KUME"
fi
kubectl config use-context "kind-$KUME" >/dev/null

echo "• KWOK kuruluyor / güncelleniyor"
kubectl apply -f "$KWOK_URL/kwok.yaml" >/dev/null
kubectl apply -f "$KWOK_URL/stage-fast.yaml" >/dev/null
kubectl -n kube-system rollout status deployment/kwok-controller --timeout=120s

for bilesen in kube-controller-manager kube-scheduler; do
  dosya="/etc/kubernetes/manifests/$bilesen.yaml"
  if docker exec "$DUGUM" grep -q -- "--kube-api-qps" "$dosya"; then
    echo "• $bilesen hız ayarı zaten yapılmış"
    continue
  fi
  echo "• $bilesen hız ayarı yapılıyor"
  # Statik pod tanımı yerinde düzenlenmez (kubelet yarım dosyayı okuyabilir): önce kopyada düzenle, sonra üstüne yaz
  docker exec "$DUGUM" sh -c "sed 's|^    - $bilesen\$|    - $bilesen\n    - --kube-api-qps=1000\n    - --kube-api-burst=2000|' $dosya > /tmp/$bilesen.yaml && cp /tmp/$bilesen.yaml $dosya"
done

echo "• Bileşenlerin yeniden başlaması bekleniyor"
sleep 15
for bilesen in kube-controller-manager kube-scheduler; do
  for _ in $(seq 1 60); do
    if kubectl -n kube-system get pod -l component=$bilesen -o jsonpath='{.items[*].spec.containers[0].command}' 2>/dev/null \
        | grep -q -- "--kube-api-qps=1000" \
       && kubectl -n kube-system wait --for=condition=Ready pod -l component=$bilesen --timeout=5s >/dev/null 2>&1; then
      echo "  ✓ $bilesen hazır (qps=1000)"
      break
    fi
    sleep 2
  done
done

kubectl get nodes
echo "Küme hazır. Demo: .venv/bin/python -m demo.calistir"
