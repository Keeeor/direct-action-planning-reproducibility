"""Matched retrained mechanism ablations; GPU training, frozen CPU evaluation.

Original data collection, architectures, objectives, seeds, optimizer, batch
size, and training lengths are retained. No test metric selects checkpoints.
"""
from __future__ import annotations
from datetime import datetime,timezone
import argparse,copy,hashlib,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from stage2_dynamic_budget.direct_action_planning_paper_evidence.data import load_development_trace_dataset
from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import load_trace_dataset
from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import FeatureNormalizer,FullTransitionNetwork
from stage2_dynamic_budget.direct_action_planning_paper_closure.models import ScaledEvidenceValueNetwork
from stage2_dynamic_budget.direct_action_planning_paper_closure.training import compute_value_target_scale
from stage2_dynamic_budget.direct_action_planning_paper_closure.calibrated_protocol import collect_calibrated_branch_dataset,evaluate_calibrated_methods
from stage2_dynamic_budget.direct_action_planning_paper_closure.control_experiment import _load_source_components
from stage2_dynamic_budget.direct_action_planning_paper_closure.planning import make_scaled_planner
torch.set_num_threads(1)
DEVICE='cuda:0'


def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(x,dtype=torch.float32):return torch.as_tensor(x,dtype=dtype,device=DEVICE)


def masked_value(train,val,seed,selected_iteration):
    scale=compute_value_target_scale(train,horizon=64)
    mask=np.ones(14,dtype=np.float32);mask[-2:]=0
    torch.manual_seed(seed+7)
    model=ScaledEvidenceValueNetwork(FeatureNormalizer.fit(train.observations),hidden_dim=64,feature_mask=mask,output_scale=scale,zero_initialize_output=True).to(DEVICE)
    optim=torch.optim.Adam(model.parameters(),lr=.001)
    x=tensor(train.observations);nxt=tensor(train.next_observations.reshape(-1,14));rew=tensor(train.rewards);allowed=tensor(train.feasible,torch.bool);alive=tensor(~train.done,torch.bool)
    vx=tensor(val.observations);vn=tensor(val.next_observations.reshape(-1,14));vr=tensor(val.rewards);vf=tensor(val.feasible,torch.bool);vd=tensor(~val.done,torch.bool)
    rng=np.random.default_rng(seed+7);history=[];chosen=None
    for iteration in range(18):
        with torch.no_grad():target=(rew+.99*alive[:,None]*model(nxt).reshape(train.n_states,4)).masked_fill(~allowed,float('-inf')).max(1).values
        losses=[]
        for _ in range(2):
            order=rng.permutation(len(x))
            for start in range(0,len(x),256):
                ix=tensor(order[start:start+256],torch.long)
                loss=nn.functional.smooth_l1_loss(model(x[ix])/scale,target[ix]/scale)
                optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();losses.append(float(loss.detach()))
        with torch.no_grad():
            vt=(vr+.99*vd[:,None]*model(vn).reshape(val.n_states,4)).masked_fill(~vf,float('-inf')).max(1).values
            mae=float((model(vx)-vt).abs().mean())
        history.append({'iteration':iteration,'training_loss':float(np.mean(losses)),'validation_bellman_mae':mae})
        if iteration==selected_iteration:chosen=copy.deepcopy(model).cpu().eval()
    if chosen is None:raise RuntimeError('frozen source-selected round missing')
    return chosen,history


def full_transition(train,val,normalizer,seed):
    torch.manual_seed(seed+2);model=FullTransitionNetwork(normalizer).to(DEVICE);optim=torch.optim.Adam(model.parameters(),lr=.001)
    def flatten(d):
        ix,a=np.nonzero(d.feasible)
        return tensor(d.observations[ix]),tensor(a,torch.long),tensor(d.next_observations[ix,a]),tensor(d.rewards[ix,a])
    x,a,y,r=flatten(train);vx,va,vy,vr=flatten(val);scale=tensor(normalizer.scale)
    rng=np.random.default_rng(seed+2);best=float('inf');best_state=None;history=[]
    for epoch in range(24):
        losses=[];order=rng.permutation(len(x))
        for start in range(0,len(x),256):
            ix=tensor(order[start:start+256],torch.long);pn,pr=model(x[ix],a[ix])
            loss=((pn-y[ix])/scale).square().mean()+nn.functional.smooth_l1_loss(pr,r[ix])
            optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();losses.append(float(loss.detach()))
        with torch.no_grad():
            pn,pr=model(vx,va);sm=float(((pn-vy)/scale).abs().mean());rm=float((pr-vr).abs().mean())
        history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_state_nmae':sm,'validation_reward_mae':rm})
        if sm+rm<best:best=sm+rm;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu().eval(),history


def blackbox_planner(model,value,weight):
    def plan(env,obs):
        scores=np.full(4,-np.inf)
        allowed=np.flatnonzero(env.action_costs<=max(env.config.budget-env.cumulative_cost,0)+1e-8)
        with torch.no_grad():
            for a in allowed:
                next_obs,reward=model(torch.as_tensor(obs,dtype=torch.float32).reshape(1,-1),torch.tensor([a]))
                future=float(value(next_obs)[0]) if weight>0 and env.t+1<env.config.horizon else 0.0
                scores[a]=float(reward[0])+.99*weight*future
        return int(np.argmax(scores)),scores
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(.15,0)
    cfg=json.loads((HERE/'protocol.json').read_text())['trace']
    cells=[(d,b,s) for d in cfg['datasets'] for b in cfg['budgets'] for s in cfg['seeds']]
    protocol={'schema':'dap.fse2027.gpu_ablation.v1','cells':cells,'methods':['dap','no_budget_horizon','blackbox_matched_weight'],'training':'original calibrated train branches:24 episodes/domain; validation_fit:6 episodes/domain; source-frozen FVI iteration and continuation weight; original Adam, batch256, hidden64, 18x2 FVI epochs; transition hidden96,24 epochs selected on validation state+reward error','evaluation':'same temporal windows as protocol.json; historically accessed test; all 100 cells; no selection on outcomes','gpu':'CUDA_VISIBLE_DEVICES=1, memory fraction 0.15, sequential cells','primary_metrics':cfg['primary_metrics'],'comparisons':['dap-no_budget_horizon','dap-blackbox_matched_weight'],'statistics':'paired training-seed aggregate, exact sign-flip, Holm across both datasets x2comparisons x4endpoints','script_sha256':sha(__file__),'source_protocol_sha256':sha(HERE/'protocol.json')}
    frozen=HERE/'gpu_ablation_protocol.json'
    if not args.smoke:
        if frozen.exists():
            if json.loads(frozen.read_text())!=protocol:raise RuntimeError('protocol drift')
        else:dump(frozen,protocol)
    else:cells=cells[:1]
    for dataset_name,budget,seed in cells:
        out=HERE/('gpu_smoke' if args.smoke else 'gpu_runs')/f'{dataset_name}__b{budget}__s{seed}'
        if (out/'manifest.json').exists():continue
        out.mkdir(parents=True,exist_ok=True);tic=time.perf_counter()
        try:
            source=ROOT/f'results/direct_action_planning_paper_closure/{cfg["source_dap_tier"]}/{dataset_name}/{cfg["source_dap_tier"]}__{dataset_name}__b{budget}__s{seed}'
            value,_,forecaster,selected=_load_source_components(source,hidden_dim=64)
            dev=load_development_trace_dataset(ROOT,dataset_name,horizon=64)
            train=collect_calibrated_branch_dataset(dev,split='train',horizon=64,budget=budget,episodes_per_domain=24,seed=seed,quantile=.95)
            val=collect_calibrated_branch_dataset(dev,split='validation_fit',horizon=64,budget=budget,episodes_per_domain=6,seed=seed+10000019,quantile=.95)
            iteration=int(selected['candidate_iteration']);iteration=17 if iteration<0 else iteration
            masked,mh=masked_value(train,val,seed,iteration)
            transition,th=full_transition(train,val,value.normalizer,seed)
            torch.save({'masked_value':masked.state_dict(),'transition':transition.state_dict(),'masked_mean':torch.tensor(masked.normalizer.mean),'masked_scale':torch.tensor(masked.normalizer.scale),'masked_output_scale':masked.output_scale,'source_selected_iteration':iteration,'source_selected_weight':selected['continuation_weight']},out/'models.pt')
            dump(out/'training.json',{'masked':mh,'transition':th,'selected':selected,'frozen_before_test_sha256':sha(out/'models.pt'),'utc':datetime.now(timezone.utc).isoformat()})
            weight=float(selected['continuation_weight'])
            methods={'dap':make_scaled_planner(value=value,forecaster=forecaster,gamma=.99,continuation_weight=weight),'no_budget_horizon':make_scaled_planner(value=masked,forecaster=forecaster,gamma=.99,continuation_weight=weight),'blackbox_matched_weight':blackbox_planner(transition,value,weight)}
            data=dev if args.smoke else load_trace_dataset(ROOT,dataset_name)
            rows,steps=evaluate_calibrated_methods(data,methods,split='validation_eval' if args.smoke else 'test',horizon=64,budget=budget,seed=seed+cfg['test_seed_offset'],episodes_per_domain=1 if args.smoke else 5,gamma=.99,quantile=.95)
            frame=pd.DataFrame(rows);frame['training_seed']=seed;frame.to_csv(out/'episodes.csv',index=False)
            pd.DataFrame(steps).to_csv(out/'steps.csv.gz',index=False,compression='gzip')
            dump(out/'manifest.json',{'status':'completed','dataset':dataset_name,'budget':budget,'training_seed':seed,'seconds':time.perf_counter()-tic,'source_checkpoint_sha256':sha(source/'models.pt'),'script_sha256':sha(__file__),'selection_used_test':False,'gpu':torch.cuda.get_device_name(0),'peak_allocated':torch.cuda.max_memory_allocated(),'artifacts':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
            print(json.dumps({'cell':[dataset_name,budget,seed],'status':'completed','seconds':time.perf_counter()-tic}),flush=True)
        except Exception:
            dump(out/'failure.json',{'traceback':traceback.format_exc(),'utc':datetime.now(timezone.utc).isoformat()});raise
    dump(HERE/('gpu_smoke_status.json' if args.smoke else 'gpu_status.json'),{'status':'completed','cells':len(cells),'utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
