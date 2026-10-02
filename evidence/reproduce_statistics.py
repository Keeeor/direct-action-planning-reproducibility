"""Recompute all completed-study paired contrasts from packaged episode data."""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixed_analysis',ROOT/'sources/analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
a.OUT=ROOT/'reproduced';a.OUT.mkdir(exist_ok=True)
counts={}
for prefix,comparators,n in [('trace',['immediate','pds_adp','mpc_8'],5000),('gpu',['no_budget_horizon','blackbox_matched_weight'],3000),('feature_control',['no_budget_horizon'],2000)]:
    frame=pd.read_csv(ROOT/'data'/f'{prefix}_episodes.csv');assert len(frame)==n
    if prefix=='feature_control':frame.loc[frame.method=='full_budget_horizon_retrained','method']='dap'
    a.paired(frame,'dataset','training_seed',comparators,a.METRICS,prefix)
    paired=a.OUT/f'{prefix}_paired.csv'
    if prefix=='feature_control':
        f=pd.read_csv(paired);f.rename(columns={'dap_mean':'full_retrain_mean','difference_dap_minus_comparator':'difference_full_retrain_minus_masked'},inplace=True);f.to_csv(paired,index=False)
        for name in ['means_by_budget','seed_means']:
            p=a.OUT/f'{prefix}_{name}.csv';f=pd.read_csv(p);f.loc[f.method=='dap','method']='full_budget_horizon_retrained';f.to_csv(p,index=False)
    for suffix in ['paired','means_by_budget','seed_means']:
        expected=pd.read_csv(ROOT/'expected'/f'{prefix}_{suffix}.csv')
        actual=pd.read_csv(a.OUT/f'{prefix}_{suffix}.csv')
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-11)
    counts[prefix]=len(pd.read_csv(paired))
frame=pd.read_csv(ROOT/'data/live_run_metrics.csv')
assert len(frame)==160
assert not frame.duplicated(['profile','budget','request_plan_seed','method']).any()
a.paired(frame,'profile','request_plan_seed',['immediate','mpc_4','dap_guard45'],['completion_rate','failure_or_slo_rate','ready_replica_seconds'],'live')
for suffix in ['paired','means_by_budget','seed_means']:
    actual=pd.read_csv(a.OUT/f'live_{suffix}.csv')
    expected=pd.read_csv(ROOT/'expected'/f'live_{suffix}.csv')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-11)
counts['live']=len(pd.read_csv(a.OUT/'live_paired.csv'))
spec=importlib.util.spec_from_file_location('schedule_check',ROOT/'sources/live_schedule_sensitivity.py')
sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
sc.DATA=a.OUT
frame.to_csv(a.OUT/'live_run_metrics.csv',index=False)
sc.main()
pd.testing.assert_frame_equal(pd.read_csv(a.OUT/'live_schedule_sensitivity_descriptive.csv'),pd.read_csv(ROOT/'expected/live_schedule_sensitivity_descriptive.csv'),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-11)
report={'verified_paired_contrasts':counts,'total_contrasts':sum(counts.values()),'numeric_tolerance':1e-10,'live_runs':160,'schedule_sensitivity_reproduced':True}
(ROOT/'reproduction_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
