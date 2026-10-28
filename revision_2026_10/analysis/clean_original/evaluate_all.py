from pathlib import Path
import argparse,json,subprocess,sys
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--gpu',type=int,required=True);p.add_argument('--execute',action='store_true')
o=p.parse_args();manifest=json.loads((R/'evaluation_manifest.json').read_text())
commands=[]
for e in manifest['entries']:
 out=R/'evaluations'/o.split/e['id']
 if out.exists():
  if not (out/'metrics.json').exists() or not (out/'predictions.npz').exists():raise RuntimeError('Incomplete existing evaluation: '+str(out))
  saved=json.loads((out/'metrics.json').read_text())
  assert saved['checkpoint_sha256']==e['checkpoint_sha256'] and saved['config_sha256']==e['config_sha256'] and saved['split']==o.split and saved['limit'] is None and saved['metrics']['n']==e['samples'][o.split]
  continue
 commands.append([sys.executable,str(R/'scripts/evaluate_one.py'),'--id',e['id'],'--split',o.split,'--gpu',str(o.gpu)])
print(json.dumps(dict(split=o.split,pending=len(commands),execute=o.execute,commands=commands),indent=2),flush=True)
if o.execute:
 checks=json.loads((R/'preflight_validation.json').read_text());assert checks['passed']
 # Clean evaluation only; validation-selected checkpoints and thresholds are immutable here.
 for cmd in commands:
  memory={int(l.split(',')[0]):int(l.split(',')[1]) for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
  if memory.get(o.gpu,999999)>=500:raise RuntimeError('GPU occupied; no job started')
  subprocess.run(cmd,check=True)
