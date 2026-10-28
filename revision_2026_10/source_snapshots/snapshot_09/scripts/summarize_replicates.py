from pathlib import Path
import json,hashlib,pickle
import numpy as np
from sklearn.metrics import f1_score
R=Path(__file__).resolve().parents[1];P=json.loads((R/'protocol/protocol.json').read_text())
OLD=Path('/data01/lyw/GLoMo_mosi_downstream_word_20261020')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
with open('/data01/lyw/GLoMo/datasets/mosi.pkl','rb') as f:examples=pickle.load(f)['dev']
labels=np.array([np.asarray(x[1]).reshape(-1)[0] for x in examples]);ids=np.array([str(x[2].decode() if isinstance(x[2],bytes) else x[2]) for x in examples])
rows={};arrays={'labels':labels,'sample_ids':ids}
for seed in [5576]+P['seeds']:
 root=OLD if seed==5576 else R;version=json.loads((root/'protocol/protocol.json').read_text())['version'];pair={}
 for variant in ['dynamic','constant']:
  d=root/'runs'/('{}_mosi_{}_seed{}'.format(version,variant,seed));m=json.loads((d/'metrics.json').read_text());h=[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()]
  assert len(h)==70 and m['test_evaluated'] is False and m['evaluation_split']=='dev'
  assert m['best_epoch']==min(range(len(h)),key=lambda i:h[i]['valid_mae'])
  assert sha(d/'best_model.pt')==m['checkpoint_sha256']
  z=np.load(d/'dev_predictions.npz');y=z['labels'];pred=z['predictions'];assert pred.shape==y.shape and len(y)==229 and np.isfinite(pred).all() and np.allclose(y,labels,atol=1e-6)
  nz=y!=0;yt=y>=0;yp=pred>=0;yn=y[nz]>0;pn=pred[nz]>0
  met=dict(mae=float(np.abs(pred-y).mean()),mse=float(np.mean((pred-y)**2)),corr=float(np.corrcoef(pred,y)[0,1]),acc7=float(np.mean(np.round(np.clip(pred,-3,3))==np.round(np.clip(y,-3,3)))),acc2=float(np.mean(yp==yt)),f1=float(f1_score(yt,yp,average='weighted')),acc2_non_zero=float(np.mean(pn==yn)),f1_non_zero=float(f1_score(yn,pn,average='weighted')))
  assert abs(met['mae']-m['valid_mae'])<1e-6
  pair[variant]={'metrics':met,'selected_epoch_one_based':m['best_epoch']+1,'checkpoint_sha256':m['checkpoint_sha256']};arrays[variant+'_seed'+str(seed)]=pred
 pair['delta_dynamic_minus_constant']={k:pair['dynamic']['metrics'][k]-pair['constant']['metrics'][k] for k in met};rows[str(seed)]=pair
summary={}
for group,seeds in [('new_replications',[42,2026]),('including_development_pilot',[5576,42,2026])]:
 out={}
 for v in ['dynamic','constant']:
  out[v]={k:{'mean':float(np.mean([rows[str(s)][v]['metrics'][k] for s in seeds])),'sample_sd':float(np.std([rows[str(s)][v]['metrics'][k] for s in seeds],ddof=1))} for k in met}
 out['paired_mae_deltas']=[rows[str(s)]['delta_dynamic_minus_constant']['mae'] for s in seeds];out['seeds']=seeds;summary[group]=out
report=dict(dataset='mosi',split='dev',pretraining_seed=5576,per_seed=rows,summary=summary,definitions='MAE primary; F1 weighted by true-label support; zero nonnegative for Acc2, omitted for nonzero metrics; each run selects earliest min dev MAE; no test evaluation.')
(R/'validation_comparison.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(R/'paired_dev_predictions.npz',**arrays)
print(json.dumps(report,indent=2))
