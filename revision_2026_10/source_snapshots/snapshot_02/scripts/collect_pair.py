from pathlib import Path
import json,time,hashlib
R=Path(__file__).resolve().parents[1]
J=json.loads((R/'jobs.json').read_text());P=json.loads((R/'protocol/protocol.json').read_text())
def save(name,value):
 f=R/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(f)
def alive(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split()[2]!='Z'
while True:
 rows={}
 for j in J:
  d=R/'runs'/('{}_'+'mosi_{}_seed5576').format(P['version'],j['variant'])
  h=d/'history.jsonl';history=[json.loads(x) for x in h.read_text().splitlines() if x.strip()] if h.exists() else []
  f=d/'metrics.json';result=json.loads(f.read_text()) if f.exists() else None
  rows[j['variant']]=dict(epochs_completed=len(history),last_epoch=history[-1] if history else None,best_valid_mae=min([x['valid_mae'] for x in history],default=None),state='complete' if result else ('running' if alive(j['pid']) else 'failed'),result=result)
 state='failed' if any(x['state']=='failed' for x in rows.values()) else ('complete' if all(x['state']=='complete' for x in rows.values()) else 'running')
 save('STATUS.json',dict(state=state,scope='MOSI seed5576 paired downstream validation; 70 epochs each; test not evaluated',runs=rows))
 if state=='complete':
  a=rows['dynamic']['result'];b=rows['constant']['result'];assert a['test_evaluated'] is False and b['test_evaluated'] is False
  save('paired_result.json',dict(dataset='mosi',seed=5576,evaluation_split='dev',dynamic=a,constant=b,delta_valid_mae_dynamic_minus_constant=a['valid_mae']-b['valid_mae'],interpretation='Negative delta favors dynamic. One-seed development comparison, not test or multi-seed estimate.'))
 if state!='running':break
 time.sleep(30)
print(state,flush=True)
