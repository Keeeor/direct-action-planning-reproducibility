"""Executable adapter fault injection with monitor-blind independent witnesses.

Faults run before scoring, before command delivery, or during accounting.
The monitor is never passed the injected fault label. This is a controlled
fault-model study, not a claim about prevalence of naturally occurring bugs.
"""
from __future__ import annotations
import copy,hashlib,inspect,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import pandas as pd
import torch
from dap.direct_action_planning_dataset_validation.data import load_trace_dataset
from dap.direct_action_planning_paper_closure.environment import make_calibrated_trace_env
from dap.direct_action_planning_paper_closure.control_experiment import _load_source_components
from dap.direct_action_planning_paper_closure.planning import make_scaled_planner
from dap.direct_action_planning_pds_adp.temporal_evaluation import load_pds_planner
from dap.direct_action_planning_pds_adp.planning import postdecision_state
torch.set_num_threads(1)


def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_hash(obs):return hashlib.sha256(np.asarray(obs,dtype=np.float32).tobytes()).hexdigest()


def execute(env,obs,previous,kind,value,forecast,weight,fault=None):
    actual=copy.deepcopy(env)
    planner_env=copy.deepcopy(previous[0] if fault=='stale_cycle_context' and previous else env)
    planner_obs=np.asarray(previous[1] if fault=='stale_cycle_context' and previous else obs)
    used=planner_env.cumulative_cost;remaining=max(planner_env.config.budget-used,0)
    allowed=planner_env.action_costs<=remaining+1e-8
    common=max(float(forecast.predict(planner_obs)),0) if kind=='dap' else None
    scores=np.full(4,-np.inf);branches={};activated=fault=='stale_cycle_context' and previous is not None
    eligible=np.flatnonzero(allowed)
    positive=[int(a) for a in eligible if planner_env.action_costs[a]>0]
    corrupt_action=positive[0] if positive else int(eligible[0])
    for action in eligible:
        a=int(action);branch=copy.deepcopy(planner_env)
        predicted=common
        if kind=='dap' and branch.t+1<branch.config.horizon:
            if fault=='branch_forecast_mismatch' and a==corrupt_action:
                predicted=common+max(1,.25*common);activated=True
            branch._arrival_rates[branch.t+1]=predicted
        if fault=='coherent_wrong_model':
            branch.capacity_deltas=branch.capacity_deltas*1.10
            activated=True
        next_obs,reward,done,trunc,info=branch.step(a)
        if kind=='pds_adp':next_obs=postdecision_state(next_obs)
        if fault=='postcost_budget_error' and a==corrupt_action and planner_env.action_costs[a]>0:
            next_obs[-2]=remaining/planner_env.config.budget;activated=True
        future=float(value.predict(next_obs.reshape(1,-1))[0]) if weight>0 and not(done or trunc) else 0
        scores[a]=reward+.99*weight*future
        branches[a]={'forecast':predicted,'post_budget':float(next_obs[-2])*planner_env.config.budget,'post_horizon':float(next_obs[-1])*planner_env.config.horizon,'cost':float(info['resource_cost']),'reward':float(reward),'score':float(scores[a])}
    selected=int(np.argmax(scores))
    if fault=='nonmaximal_selection':
        worst=min(eligible,key=lambda a:scores[a])
        if scores[selected]-scores[worst]>1e-9:selected=int(worst);activated=True
    # The simulated actuator has an explicit command mapping and a receipt.
    mapping={a:float(env.config.base_capacity+env.capacity_deltas[a]) for a in range(4)}
    alternative=next((a for a in eligible if int(a)!=selected),None)
    target=mapping[selected]
    if fault=='target_mapping_error' and alternative is not None:
        target=mapping[int(alternative)];activated=True
    delivered=target
    if fault=='delivery_override' and alternative is not None:
        delivered=mapping[int(alternative)];activated=True
    executed=min(mapping,key=lambda a:abs(mapping[a]-delivered))
    _,_,_,_,actual_info=actual.step(int(executed))
    ledger_increment=float(actual_info['resource_cost'])
    if fault=='ledger_omission' and ledger_increment>0:ledger_increment=0;activated=True
    record={'kind':kind,'step':planner_env.t,'input_hash':state_hash(planner_obs),'budget':planner_env.config.budget,'used':used,'remaining':remaining,'horizon':planner_env.config.horizon,'costs':planner_env.action_costs.tolist(),'feasible':allowed.tolist(),'common_forecast':common,'scores':scores.tolist(),'branches':branches,'selected':selected,'target':target,'mapping':mapping,'delivered':delivered,'ledger_before':env.cumulative_cost,'ledger_after':env.cumulative_cost+ledger_increment}
    # These witnesses are read from the real input and actuator-side result,
    # independently of the monitor's branch/log structures.
    witness={'step':env.t,'input_hash':state_hash(obs),'budget':env.config.budget,'used':env.cumulative_cost,'receipt_target':mapping[int(actual_info['action'])],'receipt_cost':float(actual_info['resource_cost'])}
    return record,witness,bool(activated),int(actual_info['action'])


def monitor(record,witness,level):
    if level=='ordinary_log':return []
    findings=[];r=record;a=r['selected']
    if a not in range(4) or not np.isfinite(r['scores'][a]):findings.append('valid_selection')
    if r['costs'][a]>witness['budget']-witness['used']+1e-8:findings.append('affordability')
    if r['target'] not in r['mapping'].values():findings.append('valid_target')
    if not (r['ledger_before']-1e-8<=r['ledger_after']<=r['budget']+1e-8):findings.append('ledger_bounds')
    for b in r['branches'].values():
        if not 0-1e-5<=b['post_budget']<=r['budget']+1e-5:findings.append('branch_bounds')
    if level=='standard_assertions':return list(dict.fromkeys(findings))
    if r['step']!=witness['step'] or r['input_hash']!=witness['input_hash']:findings.append('cycle_context')
    if abs(r['used']-witness['used'])>1e-8 or abs(r['remaining']-(r['budget']-witness['used']))>1e-8:findings.append('cycle_context')
    for index,b in r['branches'].items():
        if r['kind']=='dap' and abs(b['forecast']-r['common_forecast'])>1e-8:findings.append('common_forecast')
        if abs(b['post_budget']-max(r['remaining']-r['costs'][index],0))>2e-5:findings.append('postcost_budget')
        if abs(b['post_horizon']-(r['horizon']-r['step']-1))>1e-5:findings.append('cycle_context')
    if r['scores'][a]<max(r['scores'])-1e-9:findings.append('argmax')
    if r['target']!=r['mapping'][a]:findings.append('action_mapping')
    if r['target']!=witness['receipt_target']:findings.append('delivery')
    if abs(r['ledger_after']-r['ledger_before']-witness['receipt_cost'])>1e-8:findings.append('ledger_conservation')
    return list(dict.fromkeys(findings))


def main():
    cfg=json.loads((HERE/'protocol.json').read_text());tc=cfg['trace'];spec=cfg['contract_study']
    protocol={'schema':'dap.fse2027.executable_faults.v1','script_sha256':sha(__file__),'source_protocol_sha256':sha(HERE/'protocol.json'),'datasets':tc['datasets'],'budget':96,'seeds':tc['seeds'],'windows':'first registered test episode per domain','steps':[0,8,16,24,32,40,48,56],'faults':spec['faults'],'monitors':spec['monitors'],'pds_forecast_fault':'not applicable: no forecast consumed; report inactive, not a detection failure','activation':'pipeline value changed; separately report selected/executed action changes','clean':'scores and action verified against original frozen planner','generic_comparator':'standard local domain/bounds/affordability assertions, precisely enumerated in code; an equally complete cross-boundary assertion suite is equivalent to this contract checker','timing':'checker only, paired same record; excludes acquisition and persistence','utc':datetime.now(timezone.utc).isoformat()}
    dest=HERE/'fault_protocol.json'
    if dest.exists():raise RuntimeError('append-only fault protocol already exists')
    dump(dest,protocol)
    rows=[];samples=[];clean=0
    expected={'branch_forecast_mismatch':'common_forecast','postcost_budget_error':'postcost_budget','nonmaximal_selection':'argmax','target_mapping_error':'action_mapping','delivery_override':'delivery','ledger_omission':'ledger_conservation','stale_cycle_context':'cycle_context'}
    for dataset_name in tc['datasets']:
        data=load_trace_dataset(ROOT,dataset_name)
        for seed in tc['seeds']:
            source=ROOT/f'results/direct_action_planning_paper_closure/{tc["source_dap_tier"]}/{dataset_name}/{tc["source_dap_tier"]}__{dataset_name}__b96__s{seed}'
            value,_,forecast,selected=_load_source_components(source,hidden_dim=64)
            dap=make_scaled_planner(value=value,forecaster=forecast,gamma=.99,continuation_weight=selected['continuation_weight'])
            pds_source=ROOT/f'results/direct_action_planning_pds_adp/{tc["source_pds_tier"]}/{dataset_name}/{tc["source_pds_tier"]}__{dataset_name}__b96__s{seed}'
            pds,_=load_pds_planner(pds_source,gamma=.99,hidden_dim=64)
            pds_parts=inspect.getclosurevars(pds).nonlocals
            for di,domain in enumerate(data.domain_names):
                ws=seed+tc['test_seed_offset']+di*1000003
                env,_,_=make_calibrated_trace_env(data,domain,'test',horizon=64,budget=96,window_seed=ws,quantile=.95)
                obs,_=env.reset(seed=ws);previous=None
                for step in range(64):
                    if step in protocol['steps']:
                        for kind,v,w,golden in [('dap',value,selected['continuation_weight'],dap),('pds_adp',pds_parts['value'],pds_parts['weight'],pds)]:
                            base,witness,_,base_action=execute(env,obs,previous,kind,v,forecast,w)
                            action,scores=golden(env,obs)
                            if action!=base_action or not np.allclose(scores,base['scores'],atol=1e-8,rtol=1e-8):raise AssertionError('clean adapter changed planner')
                            clean+=1
                            for fault in [None,*spec['faults']]:
                                r,wi,active,executed=execute(env,obs,previous,kind,v,forecast,w,fault)
                                for level in spec['monitors']:
                                    tic=time.perf_counter_ns();findings=monitor(r,wi,level);elapsed=time.perf_counter_ns()-tic
                                    rows.append({'dataset':dataset_name,'seed':seed,'domain':domain,'step':step,'planner':kind,'fault':fault or 'clean','monitor':level,'activated':active,'detected':bool(findings),'correct_localization':expected.get(fault) in findings if fault else False,'action_changed':executed!=base_action,'monitor_microseconds':elapsed/1000,'findings':'|'.join(findings)})
                                if seed==tc['seeds'][0] and di==0 and step==0:samples.append({'fault':fault,'activated':active,'record':r,'witness':wi})
                    previous=(copy.deepcopy(env),obs.copy());a,_=dap(env,obs);obs,_,done,_,_=env.step(a)
                    if done:break
            print(dataset_name,seed,'complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'fault_results.csv',index=False)
    summary=frame.groupby(['planner','fault','monitor'],as_index=False).agg(attempts=('activated','size'),activated=('activated','sum'),detected=('detected','sum'),localized=('correct_localization','sum'),action_changed=('action_changed','sum'),median_us=('monitor_microseconds','median'))
    summary.to_csv(HERE/'fault_summary.csv',index=False)
    dump(HERE/'fault_examples.json',samples)
    dump(HERE/'fault_status.json',{'status':'completed','clean_contexts_verified':clean,'rows':len(rows),'clean_false_positives':int(frame[frame.fault=='clean'].detected.sum()),'script_sha256':sha(__file__),'utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
