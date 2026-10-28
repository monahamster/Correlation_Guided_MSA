#!/usr/bin/env python3
"""Evaluate a selected checkpoint in a prepared workspace, without training."""
import argparse,json,os,sys,pickle,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--workspace',type=Path,required=True)
 p.add_argument('--run-dir',type=Path,help='A completed new training run; omit to replay an archived checkpoint')
 p.add_argument('--checkpoint',type=Path,help='Required for archived replay; new runs must use their best_model.pt')
 p.add_argument('--split',choices=['dev','test'],default='dev')
 p.add_argument('--limit',type=int,help='Dev preflight only')
 p.add_argument('--gpu',type=int,default=0,help='Use -1 for CPU')
 p.add_argument('--output',type=Path,required=True)
 o=p.parse_args();w=o.workspace.resolve();out=o.output.resolve()
 if out.exists():p.error('Use a new output directory')
 if o.limit is not None and (o.split!='dev' or o.limit<1):p.error('Positive --limit is only allowed for dev preflight')
 rel=json.loads((w/'relocation.json').read_text());launch=json.loads((w/'launch.json').read_text())
 reg=json.loads((R/'registry.json').read_text());record=next(x for x in reg['runs'] if x['id']==rel['run'])
 run=o.run_dir.resolve() if o.run_dir else R/'runs'/rel['run']
 cfg=json.loads((run/'run_config.json').read_text());hist=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()]
 assert len(hist)==reg['epochs'][record['dataset']], 'Incomplete training history'
 selected=min(hist,key=lambda x:x['valid_mae'])['epoch_zero_based']
 if o.run_dir:
  if o.checkpoint:p.error('New-run evaluation always uses run-dir/best_model.pt')
  checkpoint=run/'best_model.pt'
 else:
  if not o.checkpoint:p.error('Provide --checkpoint for archived replay')
  checkpoint=o.checkpoint.resolve()
  assert sha(checkpoint)==record['checkpoint_sha256'],'Archived checkpoint hash mismatch'
  assert selected==record['best_epoch_zero_based']
 os.environ['CUDA_VISIBLE_DEVICES']='' if o.gpu<0 else str(o.gpu)
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 os.chdir(launch['cwd']);sys.path.insert(0,launch['cwd']);sys.argv=launch['argv'][1:]
 import main_cgmsa as m
 import numpy as np,torch
 from torch.utils.data import DataLoader
 sys.path.insert(0,str(R/'analysis/clean_original'))
 from metrics import sentiment_metrics
 # CLI comes from the prepared frozen run; compare its non-path arguments with the checkpoint's run.
 excluded={'experiment_root','corr_model_path','corr_cache_root','corr_text_encoder_path','num_classes'}
 assert all(getattr(m.args,k)==v for k,v in cfg['arguments'].items() if k not in excluded),'Run configuration does not match prepared workspace'
 m.verify_source(w);m.verify_environment(w);m.verify_inputs(w,record['dataset'])
 torch.set_num_threads(4);m.set_random_seed(record['seed'])
 ds=record['dataset'];data_path=(w/'datasets'/(ds+'.pkl')).resolve()
 with data_path.open('rb') as f:examples=pickle.load(f)[o.split]
 assert len(examples)==record['samples'][o.split]
 cache=None
 if m.args.use_fusion_correlation or m.args.use_moe_reliability:
  cache=m.strict_cache(str(data_path),ds,o.split,'../BERT_EN/',m.args.corr_cache_root,m.args.max_seq_length)[0]
 if o.limit:
  examples=examples[:o.limit]
  if cache is not None:cache={**cache,'records':cache['records'][:len(examples)]}
 data=m.get_appropriate_dataset(examples,cache);model=m.prep_for_training(1)
 model.load_state_dict(torch.load(checkpoint,map_location=m.DEVICE),strict=True);model.eval()
 if m.args.correlation_control=='constant':
  assert np.allclose(model.constant_scales.detach().cpu().numpy(),cfg['constant_scales'],atol=1e-7)
 loader=DataLoader(data,batch_size=getattr(m.args,o.split+'_batch_size'),shuffle=False,num_workers=0)
 pred=[];labels=[]
 with torch.no_grad():
  for batch in loader:
   batch=tuple(t.to(m.DEVICE) for t in batch);ids,v,a,mask,seg,y,kw=m.unpack_batch(batch)
   logits,_,_=model.test(ids,v.squeeze(1),a.squeeze(1),attention_mask=mask,token_type_ids=seg,labels=None,**kw)
   pred.extend(logits.reshape(-1).cpu().tolist());labels.extend(y.reshape(-1).cpu().tolist())
 pred=np.asarray(pred);labels=np.asarray(labels);ids=np.asarray([m.stringify_sample_id(x[2]) for x in examples])
 assert len(ids)==len(set(ids))==len(labels)
 assert np.allclose(labels,[float(np.asarray(x[1]).reshape(-1)[0]) for x in examples],atol=1e-6)
 agreement=None
 if o.split=='dev':
  with np.load(run/'dev_predictions.npz',allow_pickle=False) as z:
   assert np.array_equal(labels,z['labels'][:len(labels)])
   agreement=float(np.max(np.abs(pred-z['predictions'][:len(labels)])))
   assert np.allclose(pred,z['predictions'][:len(labels)],atol=2e-5,rtol=1e-5),agreement
 out.mkdir(parents=True)
 np.savez_compressed(out/'predictions.npz',predictions=pred,labels=labels,sample_ids=ids)
 result={'id':rel['run'],'split':o.split,'limit':o.limit,'samples':len(labels),'checkpoint_sha256':sha(checkpoint),'selected_epoch_zero_based':selected,'dev_replay_max_abs_diff':agreement,'metrics':sentiment_metrics(pred,labels) if len(labels)>1 else None,'new_training_started':False}
 (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
