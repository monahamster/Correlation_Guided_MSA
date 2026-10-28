from pathlib import Path
import sys,os,json,time,subprocess,hashlib
import numpy as np
R=Path(__file__).resolve().parent;REF=Path('/data01/lyw/GLoMo_length_pretrain_20261020')
jobs=json.loads((R/'jobs.json').read_text())['jobs'];pending=list(jobs);done=[];started=time.time()
def save(x):
 p=R/'pipeline_status.json.tmp';p.write_text(json.dumps(x,indent=2));p.replace(R/'pipeline_status.json')
save({'state':'running','diagnostics_complete':[],'remaining':[j['dataset'] for j in jobs]})
while pending:
 for j in pending[:]:
  folder=R/'runs'/(j['dataset']+'_'+j['arm']);p=folder/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
  if s.get('state')=='complete':
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
   with (R/'logs'/(folder.name+'_post.log')).open('xb') as f:ret=subprocess.run([sys.executable,str(R/'postevaluate.py'),j['dataset'],j['arm']],env=env,cwd=str(R),stdout=f,stderr=subprocess.STDOUT)
   if ret.returncode:save({'state':'failed','stage':'postdiagnostics','run':folder.name});raise SystemExit(ret.returncode)
   pending.remove(j);done.append(folder.name);save({'state':'running','diagnostics_complete':done,'remaining':[x['dataset'] for x in pending]})
  else:
   proc=Path('/proc/'+str(j['pid'])+'/stat')
   if not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[0]=='Z':save({'state':'failed','stage':'training','run':folder.name});raise SystemExit(1)
 if time.time()-started>7200:save({'state':'failed','reason':'2h timeout'});raise SystemExit(1)
 if pending:time.sleep(10)
checks={}
for ds in ['mosi','mosei']:
 a=REF/'runs'/(ds+'_same_length');b=R/'runs'/(ds+'_same_length');ai=json.loads((a/'inputs.json').read_text());bi=json.loads((b/'inputs.json').read_text())
 assert ai['initial_state_sha256']==bi['initial_state_sha256'];assert ai['train']==bi['train'] and ai['dev']==bi['dev']
 ah=json.loads((a/'history.json').read_text());bh=json.loads((b/'history.json').read_text());assert len(ah)==len(bh)==20
 for key in ['anchor_order_sha256','train_donors_sha256']:assert [x[key] for x in ah]==[x[key] for x in bh]
 aa=np.load(a/'dev_donors.npz');bb=np.load(b/'dev_donors.npz');assert set(aa.files)==set(bb.files)
 for k in aa.files:assert np.array_equal(aa[k],bb[k])
 checks[ds]={'initialization_data_orders_negative_maps_match_E2b':True,'epochs':20,'same_gpu_model':ai['gpu']==bi['gpu'],'same_physical_gpu_index':ai['visible_gpu']==bi['visible_gpu']}
for root in [R,REF]:
 for p,h in json.loads((root/'frozen_source.json').read_text()).items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h
assert (R/'scripts/pilot.py').read_bytes()==(REF/'scripts/pilot.py').read_bytes()
(R/'paired_validation.json').write_text(json.dumps({'passed':True,'checks':checks,'training_engine_identical':True,'frozen_sources_unchanged':True},indent=2))
save({'state':'complete','diagnostics_complete':done,'paired_validation':'passed','test_evaluated':False})
print('E2c PIPELINE COMPLETE',flush=True)
