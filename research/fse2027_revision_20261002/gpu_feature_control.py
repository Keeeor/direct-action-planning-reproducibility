"""Same-initialization/full-training control for the budget/horizon ablation.

The first GPU extension follows the earlier seed+7 masked-value control. This
additional arm uses exactly that seed, data, batches, loss, rounds and device,
changing only the two feature-mask entries. It retains the earlier arm and all
outcomes. Source DAP checkpoints are never changed or reselected.
"""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import json,sys,time
import numpy as np
import pandas as pd
import torch
import gpu_ablation as g
HERE=g.HERE;ROOT=g.ROOT


def main():
    torch.cuda.set_per_process_memory_fraction(.15,0)
    cfg=json.loads((HERE/'protocol.json').read_text())['trace']
    spec={'schema':'dap.fse2027.feature_control.v1','created_utc':datetime.now(timezone.utc).isoformat(),'reason':'Isolate the feature-mask effect from initialization and CPU/GPU training variation; added before inspecting paired ablation outcomes.','original_GPU_source_sha256':g.sha(HERE/'gpu_ablation.py'),'script_sha256':g.sha(__file__),'cells':100,'baseline':'full_budget_horizon_retrained','comparator':'no_budget_horizon','seed':'training_seed+7 for both arms','selection':'same original DAP frozen round and continuation scalar, no reselection','controlled':'same data, normalization, initial weights, optimization order, GPU and training length; only feature mask entries12,13 differ','primary_endpoints':['discounted_return','completion_ratio','slo_violation_rate','total_cost'],'statistics':'ten training-seed blocks; aggregate repeated budgets/windows; 20000 bootstrap draws seed20261002; exact sign flip; Holm across2datasets x4metrics'}
    lock=HERE/'feature_control_protocol.json'
    if lock.exists():
        old=json.loads(lock.read_text());spec['created_utc']=old['created_utc']
        if old!=spec:raise RuntimeError('feature-control protocol drift')
    else:g.dump(lock,spec)
    Original=g.ScaledEvidenceValueNetwork
    def full_features(*args,**kwargs):
        kwargs['feature_mask']=np.ones(14,dtype=np.float32)
        return Original(*args,**kwargs)
    for dataset_name in cfg['datasets']:
        dev=g.load_development_trace_dataset(ROOT,dataset_name,horizon=64)
        for budget in cfg['budgets']:
            for seed in cfg['seeds']:
                out=HERE/'feature_runs'/f'{dataset_name}__b{budget}__s{seed}'
                if (out/'manifest.json').exists():continue
                out.mkdir(parents=True,exist_ok=True);tic=time.perf_counter()
                source=ROOT/f'results/direct_action_planning_paper_closure/{cfg["source_dap_tier"]}/{dataset_name}/{cfg["source_dap_tier"]}__{dataset_name}__b{budget}__s{seed}'
                _,_,forecast,selected=g._load_source_components(source,hidden_dim=64)
                train=g.collect_calibrated_branch_dataset(dev,split='train',horizon=64,budget=budget,episodes_per_domain=24,seed=seed,quantile=.95)
                val=g.collect_calibrated_branch_dataset(dev,split='validation_fit',horizon=64,budget=budget,episodes_per_domain=6,seed=seed+10000019,quantile=.95)
                iteration=int(selected['candidate_iteration']);iteration=17 if iteration<0 else iteration
                with patch.object(g,'ScaledEvidenceValueNetwork',full_features):model,history=g.masked_value(train,val,seed,iteration)
                torch.save(model.state_dict(),out/'full_value.pt')
                g.dump(out/'training.json',{'history':history,'selected':selected,'frozen_before_test_sha256':g.sha(out/'full_value.pt'),'utc':datetime.now(timezone.utc).isoformat()})
                planner=g.make_scaled_planner(value=model,forecaster=forecast,gamma=.99,continuation_weight=float(selected['continuation_weight']))
                data=g.load_trace_dataset(ROOT,dataset_name)
                rows,steps=g.evaluate_calibrated_methods(data,{'full_budget_horizon_retrained':planner},split='test',horizon=64,budget=budget,seed=seed+cfg['test_seed_offset'],episodes_per_domain=5,gamma=.99,quantile=.95)
                frame=pd.DataFrame(rows);frame['training_seed']=seed;frame.to_csv(out/'episodes.csv',index=False)
                pd.DataFrame(steps).to_csv(out/'steps.csv.gz',index=False,compression='gzip')
                g.dump(out/'manifest.json',{'status':'completed','dataset':dataset_name,'budget':budget,'training_seed':seed,'seconds':time.perf_counter()-tic,'matched_masked_models_sha256':g.sha(HERE/'gpu_runs'/out.name/'models.pt'),'source_DAP_checkpoint_sha256':g.sha(source/'models.pt'),'script_sha256':g.sha(__file__)})
                print(dataset_name,budget,seed,'complete',flush=True)
    g.dump(HERE/'feature_status.json',{'status':'completed','cells':100,'utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
