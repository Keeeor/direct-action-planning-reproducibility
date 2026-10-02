"""Descriptive order-deviation check specified before the complete live matrix.

Primary analysis is unchanged. Omit the two plan seeds that had any run start
before the exact-name scheduler correction; do not choose seeds using outcomes.
No secondary hypothesis tests or significance claims are produced.
"""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/'synced/analysis'
SEEDS=[2026100201,2026100202]
METRICS=['completion_rate','failure_or_slo_rate','ready_replica_seconds']


def main():
    spec={'schema':'dap.fse2027.live_schedule_sensitivity.v1',
          'specified_utc':'2026-10-02T10:28:36Z',
          'excluded_seeds':SEEDS,'rationale':'Both plan seeds had runs started before the scheduler prefix-collision correction. Selection uses timing only, not any controller outcome.',
          'scope':'Descriptive eight-plan sensitivity; fixed ten-plan primary analysis unchanged; no new significance tests.'}
    path=DATA/'live_schedule_sensitivity_spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('specification drift')
    path.write_text(json.dumps(spec,indent=2))
    source=DATA/'live_run_metrics.csv'
    if not source.exists():return
    frame=pd.read_csv(source)
    assert len(frame)==160
    means=frame[~frame.request_plan_seed.isin(SEEDS)].groupby(['profile','request_plan_seed','method'])[METRICS].mean().reset_index()
    rows=[]
    for profile,g in means.groupby('profile'):
        for comparator in ['immediate','mpc_4','dap_guard45']:
            for metric in METRICS:
                w=g.pivot(index='request_plan_seed',columns='method',values=metric)
                assert len(w)==8 and not w.isna().any().any()
                d=w.dap-w[comparator]
                rows.append({'profile':profile,'comparator':comparator,'metric':metric,'n_pairs':8,'dap_mean':w.dap.mean(),'comparator_mean':w[comparator].mean(),'difference_dap_minus_comparator':d.mean(),'positive':int((d>1e-10).sum()),'ties':int((abs(d)<=1e-10).sum()),'negative':int((d<-1e-10).sum())})
    pd.DataFrame(rows).to_csv(DATA/'live_schedule_sensitivity_descriptive.csv',index=False)


if __name__=='__main__':main()
