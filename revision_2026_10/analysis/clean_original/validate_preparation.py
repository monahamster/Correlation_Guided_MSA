from pathlib import Path
import sys,json,subprocess,hashlib,datetime
import numpy as np
from metrics import sentiment_metrics
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'evaluation_manifest.json').read_text())
count=0
for e in manifest['entries']:
 z=np.load(Path(e['run'])/'dev_predictions.npz');m=sentiment_metrics(z['predictions'],z['labels'])
 assert abs(m['mae']-e['dev_mae'])<1e-6
 assert sum(map(sum,m['confusion2']))==m['n']
 assert sum(map(sum,m['confusion2_nonzero']))==m['n_nonzero']
 count+=1
for p in (R/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec')
dry=json.loads(subprocess.check_output([sys.executable,str(R/'scripts/evaluate_all.py'),'--split','test','--gpu','0'],text=True))
assert dry['pending']==66 and not dry['execute']
assert not (R/'evaluations/test').exists()
pre=json.loads((R/'preflight_validation.json').read_text());assert pre['passed'] and pre['count']==22 and not pre['test_evaluated']
result=dict(passed=True,saved_full_dev_predictions_scored=count,dev_prefix_model_replays=22,dev_prefix_samples=128,max_prediction_replay_difference=max(x['dev_replay_max_abs_diff'] for x in pre['results']),dry_run_test_commands=66,test_evaluated=False,remaining_training_jobs=12,remaining_training_launched=False,checked_utc=datetime.datetime.utcnow().isoformat()+'Z')
(R/'preparation_validation.json').write_text(json.dumps(result,indent=2)+'\n')
files=[p for p in (R/'scripts').glob('*.py')]+[R/x for x in ['evaluation_manifest.json','evaluation_protocol.json','remaining_training.json','execution_checklist_zh.txt']]
frozen={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/'preparation_manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
(R/'STATUS.json').write_text(json.dumps(dict(state='prepared',validation=result,next='Prepare separate frozen training snapshots and execute the 12 remaining main runs; then extend the evaluation manifest and evaluate test.'),indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
