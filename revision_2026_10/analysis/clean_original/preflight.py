from pathlib import Path
import json,sys,subprocess,concurrent.futures,datetime
R=Path(__file__).resolve().parents[1];manifest=json.loads((R/'evaluation_manifest.json').read_text())
entries=[x for x in manifest['entries'] if x['seed']==5576]
assert len(entries)==22
memory={int(l.split(',')[0]):int(l.split(',')[1]) for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
assert all(memory[g]<500 for g in [0,2]),memory
logs=R/'preflight_logs';logs.mkdir(exist_ok=False)
def worker(gpu,jobs):
 results=[]
 for e in jobs:
  command=[sys.executable,str(R/'scripts/evaluate_one.py'),'--id',e['id'],'--split','dev','--limit','128','--gpu',str(gpu)]
  with (logs/(e['id']+'.log')).open('x') as f:proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
  if proc.returncode:raise RuntimeError(e['id']+' failed; '+(logs/(e['id']+'.log')).read_text()[-2500:])
  result=json.loads((R/'evaluations/dev_preflight'/e['id']/'metrics.json').read_text());results.append(result)
  print('PREFLIGHT_PASS',e['id'],'max_diff',result['dev_replay_max_abs_diff'],flush=True)
 return results
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(worker,g,entries[i::2]) for i,g in enumerate([0,2])]
 result=[r for f in futures for r in f.result()]
report=dict(passed=True,count=len(result),samples_per_model=128,seeds=[5576],datasets=['mosi','mosei'],test_evaluated=False,finished_utc=datetime.datetime.utcnow().isoformat()+'Z',results=result)
(R/'preflight_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('ALL_PREFLIGHT_COMPLETE',len(result),flush=True)
