"""Matched frozen-checkpoint live study, isolated and append-only."""
from __future__ import annotations
import argparse
import asyncio
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['PATH']=str(HERE/'bin')+os.pathsep+os.environ.get('PATH','')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='1'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'research/direct_action_planning_k8s_prototype'))
import numpy as np
import torch
import yaml
from dap.direct_action_planning_k8s_service_repair.runtime import ServiceRepairDAPController
from dap.direct_action_planning_k8s_service_repair import audit as service_audit
from dap.direct_action_planning_k8s_service_repair import runtime_audit as service_runtime_audit
from dap.direct_action_planning_k8s_service_repair.planner import RuntimeConsistentPlanner,RuntimePlanningDecision,action_invariant_forecast
from dap.direct_action_planning_k8s_service_repair.transition import target_is_feasible
from dap.direct_action_planning_k8s_service_repair.prototype_api import ACTION_ORDER
from dap.direct_action_planning_k8s_pareto_calibration.runtime import ParetoCalibratedDAPController
from dap.direct_action_planning_k8s_robustness.connectivity import ProxyBypassedPrepare
import experiments.run_system_trial as trial
from experiments.cleanup import cleanup_target
from workload.trace_converter import build_plan,write_plan
torch.set_num_threads(1)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')


class MatchedPlanner(RuntimeConsistentPlanner):
    """Keep one transition model; vary only score horizon or feasibility reserve."""
    def __init__(self,*args,mask_guard,method,**kwargs):
        super().__init__(*args,**kwargs)
        self.mask_guard=float(mask_guard)
        self.method=method

    def feasible(self,action,ready,budget):
        return target_is_feasible(remaining_budget_seconds=budget,target_replicas=self.mapper.replicas(action),current_ready=ready,base_replicas=self.system_model.base_replicas,control_interval_seconds=self.control_interval_seconds,scale_down_guard_seconds=self.mask_guard)

    def branch(self,obs,ready,action,forecast,budget,left):
        return self.system_model.branch(observation=obs,current_ready=ready,action=action,mapper=self.mapper,forecast_arrival_rps=forecast,total_budget_seconds=self.total_budget_seconds,remaining_budget_seconds=budget,remaining_horizon_steps=left,horizon_steps=self.horizon_steps,control_interval_seconds=self.control_interval_seconds)

    def select(self,*,observation,model_ready,safety_ready,remaining_budget_seconds,remaining_horizon_steps,current_target_replicas):
        horizon=min(4,remaining_horizon_steps) if self.method=='mpc_4' else 1
        forecasts=[]
        reference=np.asarray(observation,dtype=np.float32)
        reference_ready=model_ready
        reference_budget=remaining_budget_seconds
        for depth in range(horizon):
            forecast=action_invariant_forecast(self.checkpoint,reference)
            forecasts.append(forecast)
            b=self.branch(reference,reference_ready,'no_op',forecast,reference_budget,remaining_horizon_steps-depth)
            reference=b.next_observation
            reference_ready=b.next_ready_replicas
            reference_budget=max(reference_budget-b.expected_cost_seconds,0)

        def continuation(obs,ready,budget,depth):
            if depth>=horizon:return 0.0
            values=[]
            for a in ACTION_ORDER:
                if not self.feasible(a,ready,budget):continue
                b=self.branch(obs,ready,a,forecasts[depth],budget,remaining_horizon_steps-depth)
                values.append(b.reward+self.checkpoint.gamma*continuation(b.next_observation,b.next_ready_replicas,max(budget-b.expected_cost_seconds,0),depth+1))
            return max(values)

        scores={}; feasible={};branches={}
        for a in ACTION_ORDER:
            feasible[a]=self.feasible(a,safety_ready,remaining_budget_seconds)
            if not feasible[a]:scores[a]=float('-inf');continue
            b=self.branch(np.asarray(observation,dtype=np.float32),model_ready,a,forecasts[0],remaining_budget_seconds,remaining_horizon_steps)
            branches[a]=b
            if self.method=='mpc_4':
                future=continuation(b.next_observation,b.next_ready_replicas,max(remaining_budget_seconds-b.expected_cost_seconds,0),1)
            elif self.method=='immediate' or remaining_horizon_steps<=1:
                future=0.0
            else:
                future=self.checkpoint.continuation_weight*float(self.checkpoint.value.predict(b.next_observation.reshape(1,-1))[0])
            scores[a]=float(b.reward+self.checkpoint.gamma*future)
        best=max(ACTION_ORDER,key=lambda a:scores[a])
        return RuntimePlanningDecision(action=best,greedy_action=best,target_replicas=self.mapper.replicas(best),q_values=scores,feasible=feasible,branches=branches,predicted_load_rps=forecasts[0],model_ready_replicas=model_ready,safety_ready_replicas=safety_ready,tie_retained_current_target=False)


def config_for(smoke=False):
    source=ROOT/'research/direct_action_planning_k8s_service_repair/configs/formal_v2.yaml'
    cfg=yaml.safe_load(source.read_text())
    cfg['kubernetes']['namespace']='dap-fse-revision-20261002'
    cfg['controller_defaults']['horizon_steps']=6 if smoke else 32
    cfg['paths']['system_model']=str(ROOT/'research/direct_action_planning_k8s_prototype/results/calibration/system_model.json')
    for profile,spec in cfg['profiles'].items():
        if profile=='azure_http':
            spec['checkpoint']=str(ROOT/'research/direct_action_planning_k8s_service_repair/results/checkpoints_v2/azure_http/models.pt')
        else:
            spec['checkpoint']=str(ROOT/'research/direct_action_planning_k8s_cost_calibration/results/checkpoints_v2/gentd_inference/models.pt')
    cfg['revision']={'node_port':31281,'guard_summary_sha256':sha(HERE/'guard_summary.json'),'methods':['dap','immediate','mpc_4','dap_guard45'],'mpc':'exact depth-4 enumeration, common causal no-op-reference forecast path, matched transition/cost/collector, zero terminal value','scope':'guard45 changes only the feasibility reserve, not predicted capacity or reward'}
    if not smoke:cfg['revision']['historical_test_inventory_amendment']='Only tests/direct_action_planning_k8s_service_repair/test_planner.py changed before this study; original and current hashes recorded separately; all historical runtime/model/config hashes verified exactly.'
    return cfg


def factory(method,cc,cfg):
    # One historical test file changed; every runtime/model/config file is still
    # checked exactly. Both original and current test hashes are retained.
    verify=service_audit.verify_inventory
    def verify_recorded_test_update(inventory,*,root):
        import copy
        checked=copy.deepcopy(inventory)
        if service_audit._digest_rows(checked['files'])!=checked['sha256']:raise RuntimeError('historical inventory digest invalid')
        for row in checked.get('files',[]):
            if row['path']=='tests/direct_action_planning_k8s_service_repair/test_planner.py':
                expected='sha256:19833408827365bba5f06e07d12d895913c48ddaea09876677fe3878074fcd6e'
                if 'sha256:'+sha(root/row['path'])!=expected:raise RuntimeError('unregistered test-file drift')
                row['sha256']=expected
        checked['sha256']=service_audit._digest_rows(checked['files'])
        return verify(checked,root=root)
    with patch.object(service_audit,'verify_inventory',verify_recorded_test_update),patch.object(service_runtime_audit,'verify_inventory',verify_recorded_test_update):
        if cc.profile=='azure_http':
            controller=ServiceRepairDAPController(cc,audit_contract=ROOT/'research/direct_action_planning_k8s_service_repair/contracts/development_v2_contract.json')
        else:
            controller=ParetoCalibratedDAPController(cc,audit_contract=ROOT/'research/direct_action_planning_k8s_cost_calibration/contracts/development_v2_contract.json',continuation_weight=.05,candidate_id='dap_cont_0p05')
    calibrated=json.loads((HERE/'guard_summary.json').read_text())['profiles'][cc.profile]['empirical_guard_seconds']
    guard=45.0 if method=='dap_guard45' else max(controller.system_model.scale_down_guard_seconds,calibrated)
    controller.planner=MatchedPlanner(checkpoint=controller.checkpoint,system_model=controller.system_model,mapper=cc.action_mapper,control_interval_seconds=cc.control_interval_seconds,horizon_steps=cc.horizon_steps,total_budget_seconds=cc.total_budget_seconds,tie_margin=0,mask_guard=guard,method=method)
    return controller


def prepare(kube,profile):
    if kube.namespace!='dap-fse-revision-20261002':raise RuntimeError('namespace safety check')
    cleanup_target(kube)
    kube.set_profile(profile)
    kube.scale(1)
    kube.rollout_restart()
    kube.rollout_status()
    # Wait for both Ready and desired counts to settle at base before accounting.
    deadline=time.monotonic()+45
    while True:
        status=kube.deployment_status()
        if status['ready_replicas']==1 and status['desired_replicas']==1:break
        if time.monotonic()>deadline:raise RuntimeError('initial base readiness did not settle')
        time.sleep(.25)
    return f'http://{kube.node_internal_ip()}:31281'


def plans(cfg,smoke=False):
    protocol=json.loads((HERE/'protocol.json').read_text())['live']
    seeds=[2026100299] if smoke else protocol['request_plan_seeds']
    outputs={}
    for profile,spec in cfg['profiles'].items():
        for i,seed in enumerate(seeds):
            p=HERE/('live_smoke_plans' if smoke else 'live_plans')/f'{profile}__s{seed}.jsonl'
            if not p.exists():
                rows,m=build_plan(dataset_name=spec['dataset'],domain=spec['domain'],split='validation' if smoke else 'test',horizon=cfg['controller_defaults']['horizon_steps'],interval_seconds=5,seed=seed,quantile=spec['training_quantile'],target_peak_rps=spec['target_peak_rps'],max_rps=spec['max_rps'],activity_quantile=.75 if smoke else spec['activity_quantiles'][i],project_root=ROOT)
                write_plan(rows,m,p)
            outputs[(profile,seed)]=p
    return outputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if not (HERE/'guard_summary.json').exists():raise RuntimeError('complete and freeze guard calibration first')
    cfg=config_for(args.smoke)
    cfg_path=HERE/('live_smoke.yaml' if args.smoke else 'live_config.yaml')
    content=yaml.safe_dump(cfg,sort_keys=False)
    if cfg_path.exists() and cfg_path.read_text()!=content:raise RuntimeError('config drift')
    cfg_path.write_text(content)
    generated=plans(cfg,args.smoke)
    if not args.smoke:
        inputs=[Path(__file__),HERE/'protocol.json',HERE/'guard_summary.json',cfg_path,*generated.values(),Path(cfg['paths']['system_model'])]
        inputs += [Path(spec['checkpoint']) for spec in cfg['profiles'].values()]
        inputs += list((ROOT/'src').rglob('*.py'))
        inputs += list((ROOT/'research/direct_action_planning_k8s_prototype').glob('*/*.py'))
        inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(inputs))}
        frozen=HERE/'live_frozen_inputs.json'
        if frozen.exists():
            if json.loads(frozen.read_text())['inventory']!=inventory:raise RuntimeError('frozen live input drift')
        else:dump(frozen,{'utc':datetime.now(timezone.utc).isoformat(),'inventory':inventory,'historically_accessed_trace_split':True})
    if args.prepare_only:return
    trial._make_controller=factory
    connectivity=ProxyBypassedPrepare(original_prepare=prepare)
    trial._prepare=connectivity
    protocol=json.loads((HERE/'protocol.json').read_text())['live']
    seeds=[2026100299] if args.smoke else protocol['request_plan_seeds']
    budgets=[64] if args.smoke else protocol['budgets_seconds']
    methods=['dap','mpc_4'] if args.smoke else protocol['methods']
    root=HERE/('live_smoke_runs' if args.smoke else 'live_runs');root.mkdir(exist_ok=True)
    for i,seed in enumerate(seeds):
        for profile in protocol['profiles']:
            for budget in (budgets if i%2==0 else budgets[::-1]):
                ordered=methods[i%len(methods):]+methods[:i%len(methods)]
                if len(budgets)>1 and budget==budgets[1]:ordered=ordered[::-1]
                for method in ordered:
                    name=f'{profile}__b{budget}__s{seed}__{method}'
                    prior=list(root.glob(name+'*'))
                    if any((p/'result.json').exists() and json.loads((p/'result.json').read_text()).get('status')=='completed' for p in prior):continue
                    attempt=len(prior)+1
                    if attempt>2:raise RuntimeError('maximum implementation/infrastructure retries exhausted')
                    directory=root/(name if attempt==1 else name+f'__attempt{attempt}')
                    directory.mkdir()
                    print(json.dumps({'event':'started','run':directory.name,'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
                    try:
                        with (directory/'stdout.log').open('w') as stdout,(directory/'stderr.log').open('w') as stderr,redirect_stdout(stdout),redirect_stderr(stderr):
                            result=asyncio.run(trial.run_trial(config=cfg,config_path=cfg_path,method=method,profile=profile,budget=budget,plan_path=generated[(profile,seed)],run_dir=directory,run_metadata={'revision_protocol_sha256':sha(HERE/'protocol.json'),'revision_script_sha256':sha(__file__),'new_untouched_holdout':False,'registered_pair_seed':seed,'formal_result':not args.smoke}))
                        print(json.dumps({'event':'completed','run':directory.name,'violation':result['controller']['budget_violation_seconds']}),flush=True)
                    except Exception:
                        dump(directory/'revision_failure.json',{'traceback':traceback.format_exc(),'utc':datetime.now(timezone.utc).isoformat()})
                        raise
                    finally:connectivity.close()
    dump(HERE/('live_smoke_status.json' if args.smoke else 'live_status.json'),{'status':'completed','utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
