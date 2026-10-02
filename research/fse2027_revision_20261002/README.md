# Matched-study execution sources

The frozen protocol and scripts reproduce the trace controls, learned-transition
and feature controls, executable faults, lifecycle calibration, and matched live
study. The same script snapshots are included under `evidence/sources/`.

For numerical reproduction without retraining or a cluster, run from the repository root:

```bash
python -X utf8 evidence/reproduce_statistics.py
```

This verifies all 66 paired contrasts and the scheduling sensitivity against the
complete packaged episode/run outcomes. Experiment reruns additionally require
the public trace inputs (`data/README.md`), frozen weights, and a calibrated
Kubernetes cluster. Extract `frozen/frozen_checkpoints.zip` at the repository
root to restore weights at their recorded relative paths. The protocol retains
the original paths under `PROJECT_ROOT/`; map that prefix to the checkout when
preparing destination-specific runtime configuration. Cluster context, namespace,
and networking must refer to the destination environment.

The executed studies and all planned cells are summarized in `paper/main.pdf`.
The runtime version record, selected continuation inventory, full statistical
families, lifecycle samples, and raw fault records are under `evidence/`.
