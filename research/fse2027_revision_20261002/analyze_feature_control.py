from pathlib import Path
from datetime import datetime,timezone
import json,time
import pandas as pd
import analyze as a
HERE=a.HERE;OUT=a.OUT

def main():
    while not (HERE/'feature_status.json').exists():
        if datetime.now(timezone.utc).isoformat()>'2026-10-03T08:00:00+00:00':raise RuntimeError('feature control incomplete at internal deadline')
        time.sleep(30)
    paths=sorted((HERE/'feature_runs').glob('*/episodes.csv'))
    if len(paths)!=100:raise RuntimeError('incomplete feature control')
    full=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    prior=pd.concat([pd.read_csv(p) for p in sorted((HERE/'gpu_runs').glob('*/episodes.csv'))],ignore_index=True)
    masked=prior[prior.method=='no_budget_horizon'].copy()
    keys=['dataset','budget','training_seed','domain','episode']
    match=full.merge(masked,on=keys,suffixes=('_full','_masked'),validate='one_to_one')
    assert len(match)==1000 and (match.window_start_full==match.window_start_masked).all()
    # Reuse exactly the pre-fixed statistical implementation. Rename its
    # conventional baseline alias in outputs so it cannot be confused with
    # the original, differently initialized DAP checkpoint.
    full['method']='dap'
    frame=pd.concat([full,masked],ignore_index=True)
    a.paired(frame,'dataset','training_seed',['no_budget_horizon'],a.METRICS,'feature_control')
    p=OUT/'feature_control_paired.csv';f=pd.read_csv(p)
    f.rename(columns={'dap_mean':'full_retrain_mean','difference_dap_minus_comparator':'difference_full_retrain_minus_masked'},inplace=True);f.to_csv(p,index=False)
    frame.loc[frame.method=='dap','method']='full_budget_horizon_retrained';frame.to_csv(OUT/'feature_control_episodes.csv',index=False)
    for name in ['feature_control_means_by_budget.csv','feature_control_seed_means.csv']:
        p=OUT/name;f=pd.read_csv(p);f.loc[f.method=='dap','method']='full_budget_horizon_retrained';f.to_csv(p,index=False)
    a.dump(OUT/'feature_control_integrity.json',{'status':'completed','cells':100,'paired_episodes':len(match),'same_training_initialization_and_device':True,'selected_round_and_weight_inherited_without_reselection':True,'protocol_sha256':a.sha(HERE/'feature_control_protocol.json'),'analysis_script_sha256':a.sha(__file__),'utc':datetime.now(timezone.utc).isoformat()})

if __name__=='__main__':main()
