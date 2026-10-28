from pathlib import Path
import json,subprocess,concurrent.futures,datetime,os
P=Path(__file__).resolve().parents[1]
def worker(ds,gpu):
 jobs=[(f'{ds}_{v}_seed5576','dev') for v in ['baseline','dynamic']]
 jobs += [(f'{ds}_{v}_seed{seed}','test') for seed in [42,2026,2027,3407,5576] for v in ['baseline','dynamic']]
 records=[]
 for ident,split in jobs:
  out=P/'features'/('dev_preflight' if split=='dev' else split)/ident/'metrics.json'
  if out.exists():records.append({'id':ident,'split':split,'state':'existing'});continue
  cmd=['/opt/conda/envs/glomo/bin/python',str(P/'scripts/extract_features.py'),'--id',ident,'--split',split,'--gpu',str(gpu)]
  if split=='dev':cmd+=['--limit','64']
  with (P/'logs'/f'{ident}_{split}.log').open('w') as f:
   result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  records.append({'id':ident,'split':split,'exit_code':result.returncode})
  (P/f'extraction_{ds}.json').write_text(json.dumps(records,indent=2))
  print(ds,ident,split,result.returncode,flush=True)
  if result.returncode:raise RuntimeError(ident+' failed; inspect log')
 return records
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(worker,'mosi',0),pool.submit(worker,'mosei',2)]
 allresults=[f.result() for f in fs]
(P/'extraction_complete.json').write_text(json.dumps({'state':'complete','jobs':allresults,'utc':datetime.datetime.utcnow().isoformat()},indent=2))
