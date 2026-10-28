from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];S=Path('/data01/lyw/GLoMo_main_seed_extension_20261022')
manifest=json.loads((R/'evaluation_manifest.json').read_text());assert len(manifest['entries'])==66
jobs=json.loads((S/'jobs.json').read_text());checks=json.loads((S/'paired_initialization.json').read_text())
assert len(jobs)==12 and all(j['state']=='complete' and j['exit_code']==0 for j in jobs)
assert len(checks)==12 and all(all(x.values()) for x in checks.values())
cache={}
def sha(p):
 p=Path(p).resolve();key=str(p)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
added=[]
for j in jobs:
 run=Path(j['run']);root=Path(j['root']);ds=j['dataset'];seed=j['seed'];variant=j['variant']
 cfg=json.loads((run/'run_config.json').read_text());m=json.loads((run/'metrics.json').read_text())
 h=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()];budget=70 if ds=='mosi' else 100
 assert len(h)==budget and [x['epoch_zero_based'] for x in h]==list(range(budget))
 best=min(range(budget),key=lambda i:h[i]['valid_mae'])
 assert m['best_epoch']==best and m['evaluation_split']=='dev' and not m['test_evaluated']
 assert sha(run/'best_model.pt')==m['checkpoint_sha256']
 for name,digest in cfg['source']['files'].items():assert sha(root/name)==digest,name
 for name,digest in cfg['input_sha256'].items():assert sha(name)==digest,name
 assert {k:cfg['arguments'][k] for k in manifest['common'][ds]}==manifest['common'][ds]
 z=np.load(run/'dev_predictions.npz');assert np.isfinite(z['predictions']).all() and abs(float(np.abs(z['predictions']-z['labels']).mean())-m['valid_mae'])<1e-6
 key=ds+'_'+variant+'_seed'+str(seed);member=dict(table='main',dataset=ds,seed=seed,variant=variant,id=key)
 e=dict(id=key,dataset=ds,seed=seed,run=str(run),root=str(root),memberships=[member],checkpoint=str(run/'best_model.pt'),checkpoint_sha256=m['checkpoint_sha256'],config_sha256=sha(run/'run_config.json'),source_files=cfg['source']['files'],input_sha256=cfg['input_sha256'],best_epoch_zero_based=best,epochs_completed=budget,dev_mae=m['valid_mae'],samples=cfg['samples'],source_and_input_hashes_verified=True)
 manifest['entries'].append(e);manifest['tables']['main'].append(member);added.append(key)
 print('NEW_CHECKPOINT_VERIFIED',key,flush=True)
assert len(manifest['entries'])==len({x['id'] for x in manifest['entries']})==78
assert {k:len(v) for k,v in manifest['tables'].items()}=={'main':30,'module':24,'modality':42}
assert not (R/'evaluations/test').exists()
backup=R/'evaluation_manifest_initial66.json';assert not backup.exists();shutil.copy2(R/'evaluation_manifest.json',backup)
manifest['expanded_utc']=datetime.datetime.utcnow().isoformat()+'Z';manifest['scope']='Clean evaluation of 78 frozen validation-selected checkpoints: five downstream seeds for main and three for ablations; fixed pretraining seed5576. No test-based model selection.'
(R/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'extension_inventory_validation.json').write_text(json.dumps(dict(passed=True,added=added,unique_checkpoints=78,table_rows={k:len(v) for k,v in manifest['tables'].items()},files_hashed=len(cache),test_evaluated=False),indent=2)+'\n')
p=R/'evaluation_protocol.json';protocol=json.loads(p.read_text());shutil.copy2(p,R/'evaluation_protocol_before_extension.json');protocol['existing_downstream_seeds']=[5576,42,2026,2027,3407];protocol['scope']='Clean MOSI/MOSEI evaluation of all 78 selected checkpoints; five-seed main, three-seed module and modality ablations';protocol['reuse']='78 unique checkpoints: 30 main, 24 module, 42 modality table rows; shared checkpoints evaluated once.';protocol['remaining']=[x for x in protocol['remaining'] if not x.startswith('12 planned')];p.write_text(json.dumps(protocol,indent=2)+'\n')
print('EXTENDED_INVENTORY_READY',flush=True)
