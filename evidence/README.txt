Complete matched-study statistical evidence

All studies are complete: 5,000 trace episodes, 3,000 GPU-mechanism episodes,
1,000 paired episodes from each arm of the same-initialization feature control,
640 clean executable-fault contexts, 60 lifecycle pairs, and 160 matched live runs.
The masked feature-control arm is shared with the GPU study, not a new sample.

From the repository root: python -X utf8 evidence/reproduce_statistics.py
Requires Python 3.11+, NumPy and pandas. The fixed server environment used
NumPy 2.4.4 and pandas 3.0.3. All 66 registered paired contrasts, seed/plan means,
budget means and the predefined eight-plan scheduling sensitivity are recomputed
and checked against expected tables. Ten seed/plan blocks are the primary units;
budgets and episodes are averaged within blocks. The exact analysis uses 20,000
paired bootstrap resamples, two-sided exact sign flips, and Holm families of
24 trace, 16 learned-model, eight feature-control, and 18 live endpoints.

data/: complete per-episode and per-run outcomes, raw fault results, lifecycle
samples and summary. expected/: complete contrasts, all means, integrity audits,
continuation inventory and descriptive sensitivities. protocols/ and sources/:
fixed definitions, source code for inspection, and the retained run-order amendment.
The interrupted preparation produced no request or action record; no valid run
was discarded. Temporal arrays had prior project access.

Private absolute paths in live_run_metrics.csv and two protocol files were
replaced with relative paths or PROJECT_ROOT/. Map PROJECT_ROOT/ to the local
project before inspecting runtime configurations. provenance.json records
source and derivative hashes. Numeric fields remain unchanged within CSV
precision. No private host names, keys or
credentials are included. manifest.json binds every packaged file.

This is a statistical reproduction package. Experiment reruns additionally need
the original project, trace inputs, frozen models and cluster calibration.
frozen/frozen_checkpoints.zip supplies the archived model weights; extract it at
the repository root. Public trace acquisition and preprocessing remain prerequisites
for experiment reruns. The archived request-level records were independently audited.
Its 3,390,344 request records, 5,120 decisions and all sampled ledgers were audited;
the raw-derived live outcome tables were independently reproduced locally.
paper/main.pdf is the self-contained manuscript. No separate submission attachment
is needed to read its methods, experimental conditions, and principal findings.
