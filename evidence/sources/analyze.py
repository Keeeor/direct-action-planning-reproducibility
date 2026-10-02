"""Fixed complete-matrix paired analysis and conservation/ledger audits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,itertools,json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'analysis'
METRICS=['discounted_return','completion_ratio','slo_violation_rate','total_cost']


def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readlines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def paired(frame,group,unit,comparators,metrics,prefix):
    rows=[];rng=np.random.default_rng(20261002)
    means=frame.groupby([group,unit,'method'],as_index=False)[metrics].mean()
    for dataset,sub in means.groupby(group):
        for comparator in comparators:
            for metric in metrics:
                wide=sub.pivot(index=unit,columns='method',values=metric)
                if len(wide)!=10 or wide[['dap',comparator]].isna().any().any():raise AssertionError('incomplete pairs')
                d=(wide['dap']-wide[comparator]).to_numpy();n=len(d)
                boot=d[rng.integers(0,n,size=(20000,n))].mean(1)
                permutations=(np.asarray(list(itertools.product([-1,1],repeat=n)))*d).mean(1)
                p=float(np.mean(np.abs(permutations)>=abs(d.mean())-1e-12))
                sd=float(d.std(ddof=1))
                rows.append({group:dataset,'comparator':comparator,'metric':metric,'n_pairs':n,'dap_mean':float(wide.dap.mean()),'comparator_mean':float(wide[comparator].mean()),'difference_dap_minus_comparator':float(d.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'p_exact':p,'cohens_dz':float(d.mean()/sd) if sd>1e-12 else None,'positive':int((d>1e-10).sum()),'ties':int((abs(d)<=1e-10).sum()),'negative':int((d<-1e-10).sum())})
    order=np.argsort([r['p_exact'] for r in rows]);running=0
    for rank,index in enumerate(order):
        running=max(running,min(1,rows[index]['p_exact']*(len(rows)-rank)))
        rows[index]['p_holm']=running
    pd.DataFrame(rows).to_csv(OUT/f'{prefix}_paired.csv',index=False)
    frame.groupby([group,'budget','method'],as_index=False)[metrics].mean().to_csv(OUT/f'{prefix}_means_by_budget.csv',index=False)
    means.to_csv(OUT/f'{prefix}_seed_means.csv',index=False)


def trace():
    paths=sorted((HERE/'trace_runs').glob('*/episodes.csv'))
    if len(paths)!=100:raise RuntimeError(f'trace incomplete: {len(paths)}/100')
    frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    assert len(frame)==5000
    assert not frame.duplicated(['dataset','budget','training_seed','domain','episode','method']).any()
    for _,group in frame.groupby(['dataset','budget','training_seed','domain','episode']):
        assert group.window_start.nunique()==1 and group.window_seed.nunique()==1
    frame.to_csv(OUT/'trace_episodes.csv',index=False)
    paired(frame,'dataset','training_seed',['immediate','pds_adp','mpc_8'],METRICS,'trace')
    metric=frame.groupby(['dataset','method'],as_index=False).agg(episodes=('method','size'),mean_completion=('completion_ratio','mean'),mean_clearance=('mean_step_clearance','mean'),overflow_total=('overflow','sum'),episodes_with_overflow=('overflow',lambda x:int((x>1e-8).sum())),mean_final_queue=('final_queue','mean'),mean_fifo_p95_wait_steps=('fifo_p95_wait_steps','mean'),max_conservation_residual=('conservation_residual',lambda x:float(abs(x).max())),max_overspend=('overspend','max'),mean_decision_ms=('decision_ms_mean','mean'),mean_mask_binding=('mask_binding_fraction','mean'))
    metric.to_csv(OUT/'trace_metric_audit.csv',index=False)
    selections=[]
    for p in paths:
        meta=json.loads((p.parent/'selection.json').read_text())['dap']
        first=pd.read_csv(p,nrows=1).iloc[0]
        selections.append({'dataset':first.dataset,'budget':int(first.budget),'seed':int(first.training_seed),**meta})
    pd.DataFrame(selections).to_csv(OUT/'selection_inventory.csv',index=False)
    dump(OUT/'trace_integrity.json',{'cells':len(paths),'episodes':len(frame),'steps':len(frame)*64,'maximum_mass_residual':float(abs(frame.conservation_residual).max()),'maximum_overspend':float(frame.overspend.max()),'new_untouched_holdout':False,'inputs_sha256':sha(HERE/'trace_frozen_inputs.json')})


def gpu():
    paths=sorted((HERE/'gpu_runs').glob('*/episodes.csv'))
    if len(paths)!=100:raise RuntimeError(f'GPU incomplete: {len(paths)}/100')
    frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    assert len(frame)==3000
    frame.to_csv(OUT/'gpu_episodes.csv',index=False)
    paired(frame,'dataset','training_seed',['no_budget_horizon','blackbox_matched_weight'],METRICS,'gpu')
    baseline=pd.read_csv(OUT/'trace_episodes.csv');baseline=baseline[baseline.method=='dap']
    keys=['dataset','budget','training_seed','domain','episode']
    joined=frame[frame.method=='dap'].merge(baseline,on=keys,suffixes=('_gpu','_trace'),validate='one_to_one')
    errors={m:float(abs(joined[m+'_gpu']-joined[m+'_trace']).max()) for m in METRICS}
    if max(errors.values())>1e-8:raise AssertionError(f'matched DAP baseline drift: {errors}')
    dump(OUT/'gpu_integrity.json',{'cells':100,'episodes':len(frame),'matched_DAP_episodes':len(joined),'maximum_baseline_difference':errors,'maximum_overspend':float(frame.budget_overspend.max()),'models_selected_before_test':True})


def live():
    sys.path.insert(0,str(ROOT/'research/direct_action_planning_k8s_prototype'))
    from analysis.aggregate import aggregate_run
    paths=sorted((HERE/'live_runs').glob('*/result.json'))
    if len(paths)!=160:raise RuntimeError(f'live incomplete: {len(paths)}/160')
    rows=[];integrity=[]
    for p in paths:
        folder=p.parent;r=json.loads(p.read_text());m=json.loads((folder/'run_manifest.json').read_text())
        row=aggregate_run(folder)
        request=readlines(folder/'requests.jsonl');actions=readlines(folder/'controller/controller_actions.jsonl');ledger=r['controller']['budget_ledger']
        replay=r['replay'];expected=int(replay['scheduled'])
        assert len(request)==expected==int(replay['sent'])==int(replay['completed'])+int(replay['failed'])
        assert len({x['request_id'] for x in request})==expected
        assert len(actions)==32
        total=0
        for prev,cur in zip(ledger,ledger[1:]):
            increment=max(prev['ready_replicas']-1,0)*(cur['monotonic_seconds']-prev['monotonic_seconds'])
            assert abs(increment-cur['ready_increment'])<1e-8
            total+=increment
            assert abs(total-cur['cumulative_ready_cost'])<1e-7
        assert abs(total-r['controller']['ready_cost_seconds'])<1e-7
        for a in actions:
            valid={k:v for k,v in a['q_values'].items() if a['feasible'][k]}
            assert a['action'] in valid and valid[a['action']]>=max(valid.values())-1e-8
            assert a['target_replicas']=={'no_op':1,'scale_small':2,'scale_medium':3,'scale_large':5}[a['action']]
        slo=m['slo_seconds']
        bad=sum(not (200<=int(x['http_status'])<300) or x['client_latency_seconds']>slo for x in request)
        row.update({'budget':r['budget_seconds'],'request_plan_seed':int(m['registered_pair_seed']),'failure_or_slo_rate':bad/expected,'on_time_success_ratio':1-bad/expected,'mask_binding_fraction':sum(not all(a['feasible'].values()) for a in actions)/len(actions),'maximum_sample_gap_seconds':max(s['interval_seconds'] for s in ledger),'budget_utilization':total/r['budget_seconds'],'request_dispatch_p95_lag_seconds':float(np.quantile([x['sent_offset_seconds']-x['scheduled_offset_seconds'] for x in request],.95))})
        rows.append(row)
        integrity.append({'run':folder.name,'request_count':expected,'action_count':len(actions),'ledger_samples':len(ledger),'recomputed_cost':total,'plan_sha256':m['plan_sha256'],'raw_requests_sha256':sha(folder/'requests.jsonl'),'actions_sha256':sha(folder/'controller/controller_actions.jsonl')})
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(['profile','budget','request_plan_seed','method']).any()
    for _,g in frame.groupby(['profile','request_plan_seed']):assert g.plan_sha256.nunique()==1
    frame.to_csv(OUT/'live_run_metrics.csv',index=False)
    paired(frame,'profile','request_plan_seed',['immediate','mpc_4','dap_guard45'],['completion_rate','failure_or_slo_rate','ready_replica_seconds'],'live')
    dump(OUT/'live_integrity.json',{'runs':len(frame),'requests':int(frame.total_requests.sum()),'decisions':len(frame)*32,'maximum_ledger_violation':float(frame.budget_violation_seconds.max()),'deadline_misses':int(frame.controller_deadline_misses.sum()),'all_request_counts_reconciled':True,'all_sampled_ledgers_recomputed':True,'all_feasible_argmax_mappings_verified':True,'details':integrity})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=['trace','gpu','live','all'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    spec={'schema':'dap.fse2027.analysis.v1','script_sha256':sha(__file__),'trace_primary':METRICS,'live_primary':['completion_rate','failure_or_slo_rate','ready_replica_seconds'],'live_definition':'Failure-or-SLO counts every unsuccessful HTTP request as a violation, including fast failures; measured successful-request p95 is descriptive and excludes failed requests. Original latency-only SLO rate retained separately.','primary_unit':'training seed for traces; request plan for live; budgets and episodes aggregated within units; ten paired units per dataset/profile','bootstrap':20000,'seed':20261002,'exact_sign_flip':1024,'Holm':'one family per study across datasets/profiles, comparators and primary metrics','completion_rule':'Only complete predefined matrices are analyzed; failed attempts retained independently.'}
    lock=OUT/'analysis_protocol.json'
    if lock.exists() and json.loads(lock.read_text())!=spec:raise RuntimeError('analysis specification changed')
    if not lock.exists():dump(lock,spec)
    for name,fun in [('trace',trace),('gpu',gpu),('live',live)]:
        if args.study in (name,'all'):
            fun();print(name,'complete',flush=True)
    dump(OUT/f'analysis_{args.study}_status.json',{'status':'completed','utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__)})


if __name__=='__main__':main()
