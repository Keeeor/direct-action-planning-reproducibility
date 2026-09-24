# Direct Action Planning: anonymous reproduction package

This package contains the source code, data-preprocessing code, experiment configurations, Kubernetes prototype code, and unit tests used for the finite-budget scheduling study. It is prepared for anonymous repository review.

The package intentionally excludes training logs, generated result tables and figures, model checkpoints, raw/processed dataset arrays, paper sources, historical audit reports, completed gate/lock records, and author metadata. Generated outputs are ignored by `.gitignore`.

## Layout

- `src/`: installable Python package, environments, models, planners, baselines, analysis, and evaluation code.
- `scripts/`: preprocessing, training, evaluation, diagnostics, and figure-generation entry points.
- `configs/`: synthetic and trace experiment configurations.
- `research/*/configs/` and `research/*/contracts/`: protocol configurations and execution contracts used by the paper experiments. Historical findings, audit reports, and completed result gates are omitted.
- `research/direct_action_planning_k8s_prototype/`: Kubernetes controller, workload, calibration, and deployment manifests needed by the live-control experiments.
- `data/README.md` and `data/metadata/source_manifest.json`: dataset acquisition and preprocessing instructions.
- `tests/`: self-contained unit and contract tests that do not require the omitted result artifacts.

## Environment

The package requires Python 3.11 or newer.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest
```

The default tests exercise action costs, budget accounting, environments, trace preprocessing, planning, model components, statistics, and execution-contract logic. Kubernetes integration tests require a local cluster and are not run by the default unit-test set.

## Data preprocessing

Obtain the public inputs described in `data/README.md`, place them under `data/raw/`, and run:

```bash
python scripts/preprocess_azure_trace.py
python scripts/preprocess_azure_bursty.py
python scripts/preprocess_gentd26.py
```

These commands create ignored files under `data/processed/` and write preprocessing metadata under `data/metadata/`.

## Experiments

A small synthetic smoke run is available without public trace files:

```bash
python scripts/run_experiment.py \
  --config configs/smoke.yaml \
  --method cdba \
  --budget 8 \
  --seed 0
```

Trace and Direct Action Planning entry points are available in `scripts/`. Their formal configurations are kept under `research/*/configs/`; use the configuration matching the desired protocol. Runs write only to ignored output directories such as `results/`.

The Kubernetes prototype additionally requires Docker/Kubernetes, Prometheus telemetry, and the external workload inputs. Its manifests and setup scripts are under `research/direct_action_planning_k8s_prototype/`.

## Reproduction boundary

The public Azure trace can be downloaded under its CC BY 4.0 terms. The GenTD26 raw files are not redistributed here; the source repository, pinned upstream revision, expected filenames, and preprocessing entry point are recorded in `data/metadata/source_manifest.json`. This keeps the repository anonymous and within the source dataset's redistribution terms.
