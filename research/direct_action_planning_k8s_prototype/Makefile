PYTHON ?= python
KUBE_CONTEXT ?= kind-rl-lab
KIND_CLUSTER ?= rl-lab
NAMESPACE ?= dap-k8s-prototype
IMAGE ?= dap-k8s-service:local
BASE_IMAGE ?= us-central1-docker.pkg.dev/google-samples/microservices-demo/recommendationservice:v0.10.4
PROMETHEUS_IMAGE ?= dap-prototype-prometheus:local
PROMETHEUS_BASE_IMAGE ?= quay.io/prometheus/prometheus:v2.54.1
KEDA_MANIFEST ?= kubernetes/keda/vendor/keda-2.16.1.yaml
KEDA_OPERATOR_IMAGE ?= dap-prototype-keda-operator:local
KEDA_METRICS_IMAGE ?= dap-prototype-keda-metrics:local
KEDA_ADMISSION_IMAGE ?= dap-prototype-keda-admission:local
KEDA_OPERATOR_BASE_IMAGE ?= ghcr.io/kedacore/keda:2.16.1
KEDA_METRICS_BASE_IMAGE ?= ghcr.io/kedacore/keda-metrics-apiserver:2.16.1
KEDA_ADMISSION_BASE_IMAGE ?= ghcr.io/kedacore/keda-admission-webhooks:2.16.1
KEDA_IMAGES ?= $(KEDA_OPERATOR_IMAGE) $(KEDA_METRICS_IMAGE) $(KEDA_ADMISSION_IMAGE)

.PHONY: check build cluster load-image deploy metrics-adapter prometheus-image prometheus keda-images keda calibrate smoke pilot freeze experiment analyze clean test

check:
	$(PYTHON) scripts/check_prerequisites.py --context $(KUBE_CONTEXT) --json results/prerequisites.json
	$(PYTHON) -m pytest -q tests

test:
	$(PYTHON) -m pytest -q tests

build:
	docker build --build-arg BASE_IMAGE=$(BASE_IMAGE) -t $(IMAGE) -f app/Dockerfile app

cluster:
	$(PYTHON) scripts/ensure_cluster.py --context $(KUBE_CONTEXT) --kind-cluster $(KIND_CLUSTER)

load-image:
	$(PYTHON) scripts/load_image.py --image $(IMAGE) --context $(KUBE_CONTEXT) --kind-cluster $(KIND_CLUSTER)

deploy:
	kubectl --context $(KUBE_CONTEXT) apply -f kubernetes/namespace.yaml
	kubectl --context $(KUBE_CONTEXT) apply -f kubernetes/rbac.yaml
	kubectl --context $(KUBE_CONTEXT) apply -f kubernetes/deployment.yaml
	kubectl --context $(KUBE_CONTEXT) apply -f kubernetes/service.yaml
	kubectl --context $(KUBE_CONTEXT) -n $(NAMESPACE) rollout status deployment/dap-worker --timeout=180s

metrics-adapter: build load-image
	$(PYTHON) scripts/ensure_metrics_adapter_tls.py --context $(KUBE_CONTEXT) --namespace $(NAMESPACE)
	kubectl --context $(KUBE_CONTEXT) apply -f kubernetes/metrics-server/metrics-server.yaml
	kubectl --context $(KUBE_CONTEXT) -n $(NAMESPACE) rollout restart deployment/dap-prototype-metrics-server
	kubectl --context $(KUBE_CONTEXT) -n $(NAMESPACE) rollout status deployment/dap-prototype-metrics-server --timeout=180s
	kubectl --context $(KUBE_CONTEXT) get --raw /apis/metrics.k8s.io/v1beta1/namespaces/$(NAMESPACE)/pods

prometheus-image:
	docker build --build-arg BASE_IMAGE=$(PROMETHEUS_BASE_IMAGE) -t $(PROMETHEUS_IMAGE) -f kubernetes/prometheus/Dockerfile kubernetes/prometheus

prometheus: prometheus-image
	$(PYTHON) scripts/load_image.py --image $(PROMETHEUS_IMAGE) --context $(KUBE_CONTEXT) --kind-cluster $(KIND_CLUSTER)
	kubectl --context $(KUBE_CONTEXT) apply -k kubernetes/prometheus
	kubectl --context $(KUBE_CONTEXT) -n $(NAMESPACE) rollout status deployment/prometheus --timeout=180s

keda-images:
	docker build --build-arg BASE_IMAGE=$(KEDA_OPERATOR_BASE_IMAGE) -t $(KEDA_OPERATOR_IMAGE) -f kubernetes/keda/Dockerfile kubernetes/keda
	docker build --build-arg BASE_IMAGE=$(KEDA_METRICS_BASE_IMAGE) -t $(KEDA_METRICS_IMAGE) -f kubernetes/keda/Dockerfile kubernetes/keda
	docker build --build-arg BASE_IMAGE=$(KEDA_ADMISSION_BASE_IMAGE) -t $(KEDA_ADMISSION_IMAGE) -f kubernetes/keda/Dockerfile kubernetes/keda

keda: prometheus keda-images
	$(PYTHON) scripts/load_images.py --context $(KUBE_CONTEXT) --kind-cluster $(KIND_CLUSTER) $(KEDA_IMAGES)
	kubectl --context $(KUBE_CONTEXT) apply --server-side --force-conflicts -f $(KEDA_MANIFEST)
	kubectl --context $(KUBE_CONTEXT) -n keda rollout status deployment/keda-operator --timeout=240s
	kubectl --context $(KUBE_CONTEXT) -n keda rollout status deployment/keda-metrics-apiserver --timeout=240s
	kubectl --context $(KUBE_CONTEXT) -n keda rollout status deployment/keda-admission --timeout=240s
	kubectl --context $(KUBE_CONTEXT) get crd scaledobjects.keda.sh

calibrate:
	$(PYTHON) calibration/capacity_calibration.py --config experiments/configs/calibration.yaml
	$(PYTHON) calibration/startup_delay_calibration.py --config experiments/configs/calibration.yaml
	$(PYTHON) calibration/fit_system_model.py --input results/calibration --output results/calibration/system_model.json

smoke:
	$(PYTHON) experiments/run_single.py --config experiments/configs/smoke.yaml

pilot:
	$(PYTHON) experiments/run_matrix.py --config experiments/configs/pilot.yaml

freeze:
	$(PYTHON) experiments/freeze_protocol.py --config experiments/configs/formal.yaml

experiment:
	$(PYTHON) experiments/run_matrix.py --config experiments/configs/formal.yaml --require-frozen

analyze:
	$(PYTHON) analysis/aggregate.py --results results/runs --output results/analysis
	$(PYTHON) analysis/pareto.py --input results/analysis/run_metrics.csv --output results/analysis
	$(PYTHON) analysis/statistics.py --input results/analysis/run_metrics.csv --output results/analysis
	$(PYTHON) analysis/plots.py --input results/analysis --output results/figures

clean:
	$(PYTHON) experiments/cleanup.py --context $(KUBE_CONTEXT) --namespace $(NAMESPACE)
