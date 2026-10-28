from pathlib import Path
import json,hashlib,pickle
import numpy as np
from sklearn.metrics import f1_score
R=Path(__file__).resolve().parents[1];entries=json.loads((R/'run_matrix.json').read_text());rows={};arrays={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
with open('/data01/lyw/GLoMo/datasets/mosei.pkl','rb') as f:examples=pickle.load(f)['dev']
yref=np.array([np.asarray(x[1]).reshape(-1)[0] for x in examples]);arrays['labels']=yref;arrays['sample_ids']=np.array([str(x[2].decode() if isinstance(x[2],bytes) else x[2]) for x in examples])
for seed in [5576,42,2026]:
 rows[str(seed)]={}
 for e in entries:
  D=Path(e['pilot_root'] if seed==5576 else e['root']);p=json.loads((D/'protocol/protocol.json').read_text());pair={}
  for variant in ['dynamic','constant']:
   d=D/'runs'/('{}_mosei_{}_seed{}'.format(p['version'],variant,seed));m=json.loads((d/'metrics.json').read_text());h=[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
   assert len(h)==100 and m['evaluation_split']=='dev' and m['test_evaluated'] is False
   assert m['best_epoch']==min(range(100),key=lambda i:h[i]['valid_mae']);assert sha(d/'best_model.pt')==m['checkpoint_sha256']
   z=np.load(d/'dev_predictions.npz');pred=z['predictions'];y=z['labels'];assert pred.shape==y.shape and np.isfinite(pred).all() and np.allclose(y,yref,atol=1e-6)
   nz=y!=0;yt=y>=0;yp=pred>=0;yn=y[nz]>0;pn=pred[nz]>0
   metrics=dict(mae=float(np.abs(pred-y).mean()),mse=float(np.mean((pred-y)**2)),corr=float(np.corrcoef(pred,y)[0,1]),acc7=float(np.mean(np.round(np.clip(pred,-3,3))==np.round(np.clip(y,-3,3)))),acc2=float(np.mean(yp==yt)),f1=float(f1_score(yt,yp,average='weighted')),acc2_non_zero=float(np.mean(pn==yn)),f1_non_zero=float(f1_score(yn,pn,average='weighted')))
   assert abs(metrics['mae']-m['valid_mae'])<1e-6
   pair[variant]=dict(metrics=metrics,selected_epoch_one_based=m['best_epoch']+1,checkpoint_sha256=m['checkpoint_sha256']);arrays['seed{}_{}_{}'.format(seed,e['readout'],variant)]=pred
  pair['delta_dynamic_minus_constant']={k:pair['dynamic']['metrics'][k]-pair['constant']['metrics'][k] for k in metrics};rows[str(seed)][e['readout']]=pair
summary={}
for name,seeds in [('new_replications',[42,2026]),('including_development_pilot',[5576,42,2026])]:
 out={'seeds':seeds,'metrics':{},'paired_deltas':{}}
 for readout in ['sentence','word']:
  out['metrics'][readout]={v:{k:{'mean':float(np.mean([rows[str(s)][readout][v]['metrics'][k] for s in seeds])),'sample_sd':float(np.std([rows[str(s)][readout][v]['metrics'][k] for s in seeds],ddof=1))} for k in metrics} for v in ['dynamic','constant']}
  out['paired_deltas'][readout+'_dynamic_minus_constant']={str(s):rows[str(s)][readout]['delta_dynamic_minus_constant']['mae'] for s in seeds}
 for v in ['dynamic','constant']:out['paired_deltas'][v+'_word_minus_sentence']={str(s):rows[str(s)]['word'][v]['metrics']['mae']-rows[str(s)]['sentence'][v]['metrics']['mae'] for s in seeds}
 summary[name]=out
report=dict(dataset='mosei',split='dev',fixed_pretraining_seed=5576,pretraining_objective='sentence',per_seed=rows,summary=summary,definitions='Each run selects earliest minimum dev MAE; no test. F1 weighted by true-label support, zero nonnegative in Acc2, excluded in nonzero metrics. Downstream seeds vary; frozen pretraining checkpoint unchanged.')
(R/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(R/'dev_predictions.npz',**arrays);print(json.dumps(summary,indent=2))
