from pathlib import Path
import json,time,subprocess,sys
R=Path(__file__).resolve().parents[1]
J=json.loads((R/'jobs.json').read_text());P=json.loads((R/'protocol/protocol.json').read_text())
def save(name,value):
 f=R/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(f)
def alive(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split()[2]!='Z'
while True:
 rows={};configs={}
 for j in J:
  key='{}_seed{}'.format(j['variant'],j['seed']);d=R/'runs'/('{}_mosi_{}'.format(P['version'],key))
  h=d/'history.jsonl';history=[]
  if h.exists():
   for line in h.read_text().splitlines():
    try:history.append(json.loads(line))
    except json.JSONDecodeError:break
  f=d/'metrics.json';result=json.loads(f.read_text()) if f.exists() else None
  c=d/'run_config.json'
  if c.exists():configs[key]=json.loads(c.read_text())
  rows[key]=dict(epochs_completed=len(history),best_valid_mae=min([x['valid_mae'] for x in history],default=None),state='complete' if result else ('running' if alive(j['pid']) else 'failed'),result=result)
 pairs={}
 for seed in P['seeds']:
  a=configs.get('dynamic_seed'+str(seed));b=configs.get('constant_seed'+str(seed))
  if a and b:
   keys=['backbone_initialization_sha256','routing_initialization_sha256','input_sha256','samples']
   checks={k:a[k]==b[k] for k in keys};checks['constant_scales']=a['constant_scales'] is None and max(abs(x-y) for x,y in zip(b['constant_scales'],[1.0057005882263184,1.0066330432891846,1.0070469379425049]))<2e-6;assert all(checks.values()),checks
   pairs[str(seed)]={'checks':checks,'backbone':a[keys[0]],'routing':a[keys[1]],'loader_seed':seed+10000,'training_rng_seed':seed+40000}
 if pairs:save('protocol/paired_initialization.json',pairs)
 state='failed' if any(x['state']=='failed' for x in rows.values()) else ('complete' if all(x['state']=='complete' for x in rows.values()) else 'running')
 save('STATUS.json',dict(state=state,scope='MOSI downstream seeds42,2026; 70 epochs; frozen correlation seed5576; dev only',runs=rows,paired_initialization_checked=list(pairs)))
 if state=='complete':
  subprocess.run([sys.executable,str(R/'scripts/summarize_replicates.py')],check=True)
 if state!='running':break
 time.sleep(30)
print(state,flush=True)
