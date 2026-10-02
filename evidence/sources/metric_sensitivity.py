"""Descriptive aggregation sensitivity prompted by the conservation audit.

This does not replace the fixed primary per-episode completion endpoint.
It adds no new significance claims or policy selection.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
HERE=Path(__file__).resolve().parent
DATA=HERE/'synced/analysis'
f=pd.read_csv(DATA/'trace_episodes.csv');rows=[]
for (dataset,method),g in f.groupby(['dataset','method']):
    arrivals=g.total_arrivals.sum()
    row={'dataset':dataset,'method':method,'episodes':len(g),'zero_arrival_episodes':int((g.total_arrivals==0).sum()),'macro_completion':g.completion_ratio.mean(),'nonempty_macro':g[g.total_arrivals>0].completion_ratio.mean(),'request_weighted_completion':g.total_served.sum()/arrivals,'request_weighted_drop':g.overflow.sum()/arrivals,'request_weighted_unfinished':g.final_queue.sum()/arrivals}
    assert abs(row['request_weighted_completion']+row['request_weighted_drop']+row['request_weighted_unfinished']-1)<1e-10
    rows.append(row)
pd.DataFrame(rows).to_csv(DATA/'metric_sensitivity_descriptive.csv',index=False)
(DATA/'metric_sensitivity_spec.json').write_text(json.dumps({'status':'completed','scope':'Exploratory descriptive aggregation audit; primary endpoint unchanged; no inferential claims.','trigger':'Conservation audit identified zero-demand episodes and clipped queue overflow.','source_sha256':hashlib.sha256((DATA/'trace_episodes.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'utc':datetime.now(timezone.utc).isoformat()},indent=2))
