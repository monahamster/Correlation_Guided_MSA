from pathlib import Path
import argparse,json,os,sys,hashlib,datetime,pickle
R=Path('/data01/lyw/GLoMo_final_evaluation_20261022')
sys.path.insert(0,str(R/'scripts'))
p=argparse.ArgumentParser()
p.add_argument('--id',required=True);p.add_argument('--split',choices=['dev','test'],default='dev')
p.add_argument('--gpu',type=int,required=True);p.add_argument('--limit',type=int)
p.add_argument('--output-root',type=Path,default=Path('/data01/lyw/GLoMo/revision_plan_20261020/p6_analysis_visuals_20261026/features'))
o=p.parse_args()
if o.limit is not None and (o.split!='dev' or o.limit<=0):p.error('--limit is only for positive-sized dev preflight')
manifest=json.loads((R/'evaluation_manifest.json').read_text())
e=next(x for x in manifest['entries'] if x['id']==o.id)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
root=Path(e['root']);run=Path(e['run'])
assert sha(run/'run_config.json')==e['config_sha256']
assert sha(e['checkpoint'])==e['checkpoint_sha256']
for name,digest in e['source_files'].items():assert sha(root/name)==digest,name
cfg=json.loads((run/'run_config.json').read_text());a=cfg['arguments'];ds=e['dataset']
assert not a['load'] and not a['paired_init_load'] and not a['paired_init_save']
assert not a['test_audio_noise_std'] and not a['test_visual_mask_ratio']
data_path=(root/'datasets'/(ds+'.pkl')).resolve()
for path in [data_path,(root/'BERT_EN/pytorch_model.bin').resolve()]:
 assert sha(path)==e['input_sha256'][str(path)],str(path)
os.environ['CUDA_VISIBLE_DEVICES']=str(o.gpu)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,torch
from torch.utils.data import DataLoader
from metrics import sentiment_metrics
torch.set_num_threads(4)
os.chdir(root/'Correlation_Guided_MSA');sys.path.insert(0,str(Path.cwd()))
sys.argv=[sys.argv[0],'--dataset',ds,'--VISUAL_DIM',str(a['VISUAL_DIM']),'--d_l',str(a['d_l']),'--max_seq_length',str(a['max_seq_length'])]
import main_cgmsa as m
m.args=argparse.Namespace(**a);m.set_random_seed(e['seed'])
assert m.DEVICE.type=='cuda','GPU expected'
with data_path.open('rb') as f:examples=pickle.load(f)[o.split]
cache=None
if a['use_fusion_correlation'] or a['use_moe_reliability']:
 cache=m.strict_cache(str(data_path),ds,o.split,'../BERT_EN/',a['corr_cache_root'],a['max_seq_length'])[0]
assert len(examples)==e['samples'][o.split]
if o.limit:
 examples=examples[:o.limit]
 if cache is not None:cache={**cache,'records':cache['records'][:len(examples)]}
data=m.get_appropriate_dataset(examples,cache)
model=m.prep_for_training(1)
model.load_state_dict(torch.load(e['checkpoint'],map_location=m.DEVICE),strict=True)
model.eval()
if a['correlation_control']=='constant':
 assert torch.isfinite(model.constant_scales).all()
 assert np.allclose(model.constant_scales.detach().cpu().numpy(),cfg['constant_scales'],atol=1e-7)
loader=DataLoader(data,batch_size=a[o.split+'_batch_size'],shuffle=False,num_workers=0)
pred=[];labels=[];features=[]
with torch.no_grad():
 for batch in loader:
  batch=tuple(t.to(m.DEVICE) for t in batch)
  ids,v,au,mask,seg,y,kw=m.unpack_batch(batch)
  logits,_,reprs=model.test(ids,v.squeeze(1),au.squeeze(1),attention_mask=mask,token_type_ids=seg,labels=None,**kw)
  features.append(reprs.detach().cpu().numpy())
  pred.extend(logits.reshape(-1).cpu().tolist());labels.extend(y.reshape(-1).cpu().tolist())
pred=np.asarray(pred);labels=np.asarray(labels)
ids=np.asarray([m.stringify_sample_id(x[2]) for x in examples])
assert len(ids)==len(set(ids))==len(labels)
assert np.allclose(labels,[float(np.asarray(x[1]).reshape(-1)[0]) for x in examples],atol=1e-6)
features=np.concatenate(features,axis=0);assert np.isfinite(features).all()
metrics=sentiment_metrics(pred,labels);agreement=None
if o.split=='test':
 saved=np.load(R/'evaluations/test'/o.id/'predictions.npz')
 assert np.array_equal(ids,saved['sample_ids']) and np.array_equal(labels,saved['labels'])
 agreement=float(np.max(np.abs(pred-saved['predictions'])))
 assert np.allclose(pred,saved['predictions'],atol=2e-5,rtol=1e-5),(o.id,agreement)
if o.split=='dev':
 saved=np.load(run/'dev_predictions.npz');n=len(labels)
 assert np.array_equal(labels,saved['labels'][:n])
 agreement=float(np.max(np.abs(pred-saved['predictions'][:n])))
 assert np.allclose(pred,saved['predictions'][:n],atol=2e-5,rtol=1e-5),(o.id,agreement)
out=o.output_root/(o.split+'_preflight' if o.limit else o.split)/o.id
out.mkdir(parents=True,exist_ok=False)
np.savez_compressed(out/'predictions.npz',predictions=pred,labels=labels,sample_ids=ids,features=features)
result=dict(id=o.id,dataset=ds,seed=e['seed'],split=o.split,limit=o.limit,metrics=metrics,selected_epoch_zero_based=e['best_epoch_zero_based'],checkpoint=e['checkpoint'],checkpoint_sha256=e['checkpoint_sha256'],config_sha256=e['config_sha256'],dev_replay_max_abs_diff=agreement,test_evaluated=o.split=='test',created_utc=datetime.datetime.utcnow().isoformat()+'Z',constant_scales=cfg['constant_scales'],selection='Existing earliest minimum dev MAE; no model or threshold selection in evaluator')
(out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('EVALUATION_COMPLETE',o.id,o.split,len(labels),'dev_replay_max_diff',agreement,flush=True)
