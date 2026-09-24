#!/usr/bin/env bash
set -euo pipefail

echo "Core: install Docker, kubectl, Python 3.11+, and either Minikube or kind."
echo "Python: python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
echo "Optional k6: follow https://grafana.com/docs/k6/latest/set-up/install-k6/"
echo "Optional KEDA: follow https://keda.sh/docs/latest/deploy/"
echo "This script is informational and does not modify the host."

