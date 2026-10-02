"""Correct exact run-id matching without changing any frozen controller code.

The original scheduler's glob('...__dap*') also matched dap_guard45 and skipped
some DAP cells. This wrapper restricts completion checks to the exact run id or
its numbered attempt directories. Existing valid outcomes remain untouched.
"""
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timezone
import json,re,hashlib
import live_evaluate as live
HERE=live.HERE


def main():
    root=HERE/'live_runs'
    original_glob=Path.glob
    def exact_run_attempts(self,pattern,*args,**kwargs):
        paths=original_glob(self,pattern,*args,**kwargs)
        if self==root and isinstance(pattern,str) and pattern.endswith('*'):
            name=pattern[:-1]
            return (p for p in paths if p.name==name or re.fullmatch(re.escape(name)+r'__attempt[2-9][0-9]*',p.name))
        return paths
    amendment=HERE/'live_schedule_amendment.json'
    if not amendment.exists():
        completed=sorted(p.parent.name for p in original_glob(root,'*/result.json'))
        incomplete=sorted(p.name for p in root.iterdir() if p.is_dir() and not (p/'result.json').exists())
        live.dump(amendment,{'schema':'dap.fse2027.scheduler_fix.v1','utc':datetime.now(timezone.utc).isoformat(),'cause':'Prefix collision: dap completion checks also matched dap_guard45.','fix':'Exact run-id or numbered-attempt matching; no policy/model/metric/budget/plan changes.','wrapper_sha256':live.sha(__file__),'original_runner_sha256':live.sha(HERE/'live_evaluate.py'),'completed_before_fix':completed,'incomplete_before_fix':incomplete,'order_deviation':'Missing cells are filled at the beginning of resumed schedule. Report the deviation and a descriptive sensitivity omitting pre-fix plan seeds.','retain_all_existing_runs':True})
    else:
        spec=json.loads(amendment.read_text())
        if spec['wrapper_sha256']!=live.sha(__file__) or spec['original_runner_sha256']!=live.sha(HERE/'live_evaluate.py'):raise RuntimeError('scheduler amendment drift')
    original_trial=live.trial.run_trial
    async def recorded_trial(*args,**kwargs):
        metadata=dict(kwargs.get('run_metadata') or {})
        metadata['scheduler_amendment_sha256']=live.sha(amendment)
        kwargs['run_metadata']=metadata
        return await original_trial(*args,**kwargs)
    with patch.object(Path,'glob',exact_run_attempts),patch.object(live.trial,'run_trial',recorded_trial):
        live.main()


if __name__=='__main__':main()
