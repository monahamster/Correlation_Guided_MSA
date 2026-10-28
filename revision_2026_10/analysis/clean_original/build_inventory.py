from pathlib import Path
import json,hashlib,datetime
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
mod=read('/data01/lyw/GLoMo_module_ablation_20261021/comparison.json')
modal=read('/data01/lyw/GLoMo_modality_ablation_20261021/comparison.json')
entries={};tables={'main':[],'module':[],'modality':[]}
def add(table,ds,seed,variant,path):
 path=Path(path);key=ds+'_'+variant+'_seed'+str(seed)
 identity=str(path.resolve())
 if identity not in entries:entries[identity]=dict(id=key,dataset=ds,seed=seed,run=str(path),root=str(path.parents[1]),memberships=[])
 item=entries[identity];member=dict(table=table,dataset=ds,seed=seed,variant=variant,id=item['id'])
 item['memberships'].append(member);tables[table].append(member)
for ds in ['mosi','mosei']:
 for seed in [5576,42,2026]:
  rows=mod['per_seed'][ds][str(seed)]
  for v in ['baseline','constant','dynamic']:
   path=rows['baseline']['run'] if v=='baseline' else rows['full']['run']
   if v=='constant':path=path.replace('_dynamic_seed','_constant_seed')
   add('main',ds,seed,v,path)
  for v in ['baseline','fusion_only','moe_only','full']:add('module',ds,seed,v,rows[v]['run'])
  for v,row in modal['per_seed'][ds][str(seed)].items():add('modality',ds,seed,v,row['run'])
cache={}
def sha(p):
 p=Path(p).resolve();key=str(p)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
result=[];paired={};common={}
for i,e in enumerate(entries.values()):
 run=Path(e['run']);root=Path(e['root']);cfg=read(run/'run_config.json');m=read(run/'metrics.json');args=cfg['arguments']
 hist=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines() if x.strip()]
 budget=70 if e['dataset']=='mosi' else 100
 assert len(hist)==budget and [x['epoch_zero_based'] for x in hist]==list(range(budget)),e['id']
 best=min(range(budget),key=lambda k:hist[k]['valid_mae'])
 assert m['best_epoch']==best and m['evaluation_split']=='dev' and not m['test_evaluated']
 assert abs(m['valid_mae']-hist[best]['valid_mae'])<1e-10
 assert sha(run/'best_model.pt')==m['checkpoint_sha256']
 for path,digest in cfg['source']['files'].items():assert sha(root/path)==digest,(e['id'],path)
 for path,digest in cfg['input_sha256'].items():assert sha(path)==digest,(e['id'],path)
 z=np.load(run/'dev_predictions.npz');assert np.isfinite(z['predictions']).all()
 assert abs(float(np.abs(z['predictions']-z['labels']).mean())-m['valid_mae'])<1e-6
 sharedkey=(e['dataset'],e['seed']);digest=cfg['backbone_initialization_sha256']
 assert paired.setdefault(sharedkey,digest)==digest,(e['id'],'initialization')
 dimensions={k:args[k] for k in ['TEXT_DIM','ACOUSTIC_DIM','VISUAL_DIM','max_seq_length','n_epochs','d_l','layers','learning_rate','train_batch_size']}
 assert common.setdefault(e['dataset'],dimensions)==dimensions,(e['id'],'dataset hyperparameters')
 e.update(checkpoint=str(run/'best_model.pt'),checkpoint_sha256=m['checkpoint_sha256'],config_sha256=sha(run/'run_config.json'),source_files=cfg['source']['files'],input_sha256=cfg['input_sha256'],best_epoch_zero_based=best,epochs_completed=budget,dev_mae=m['valid_mae'],samples=cfg['samples'],source_and_input_hashes_verified=True)
 result.append(e)
 if (i+1)%6==0:print('VERIFIED',i+1,'/',len(entries),flush=True)
assert len(result)==66 and {k:len(v) for k,v in tables.items()}=={'main':18,'module':24,'modality':42}
manifest=dict(created_utc=datetime.datetime.utcnow().isoformat()+'Z',scope='Frozen selected-checkpoint clean evaluation inventory. Three existing downstream seeds; fixed pretraining seed5576. Dev-screened settings, not new independent training replications.',entries=result,tables=tables,common=common,test_evaluated=False)
(R/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
missing=[dict(dataset=ds,variant=v,seed=s,n_epochs=70 if ds=='mosi' else 100,state='planned') for ds in ['mosi','mosei'] for s in [2027,3407] for v in ['baseline','constant','dynamic']]
(R/'remaining_training.json').write_text(json.dumps(dict(reason='Complete the previously planned five-seed main comparison; reviewer asks 3-5 seeds, existing three seeds already meet its lower bound.',pretraining_retrained=False,jobs=missing),indent=2)+'\n')
(R/'inventory_validation.json').write_text(json.dumps(dict(passed=True,unique_checkpoints=len(result),table_rows={k:len(v) for k,v in tables.items()},files_hashed=len(cache),existing_seeds=[5576,42,2026],planned_main_seeds=[5576,42,2026,2027,3407],missing_main_runs=len(missing),paired_backbone_checks=True,earliest_dev_mae_selection_verified=True,test_evaluated=False),indent=2)+'\n')
print('INVENTORY_COMPLETE',len(result),'models;',len(missing),'planned extra runs',flush=True)
