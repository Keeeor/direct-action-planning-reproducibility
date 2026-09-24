#!/usr/bin/env bash
set -euo pipefail

context="${KUBE_CONTEXT:-kind-rl-lab}"
namespace="${NAMESPACE:-dap-k8s-prototype}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

kubectl --context "$context" apply -f "$root/kubernetes/metrics-server/metrics-server.yaml"
kubectl --context "$context" -n "$namespace" rollout status deployment/dap-prototype-metrics-server --timeout=180s
kubectl --context "$context" get --raw /apis/metrics.k8s.io/v1beta1/nodes

# KEDA is an optional, cluster-scoped operator. The pinned installer is only
# attempted after Metrics Server succeeds; it never silently substitutes an
# emulated controller when the CRD/operator cannot be installed.
keda_url="https://github.com/kedacore/keda/releases/download/v2.18.1/keda-2.18.1-core.yaml"
kubectl --context "$context" apply -f "$keda_url"
kubectl --context "$context" get crd scaledobjects.keda.sh
