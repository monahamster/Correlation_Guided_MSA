from pathlib import Path
import json,hashlib,pickle
import numpy as np
from sklearn.metrics import f1_score
R=Path(__file__).resolve().parents[1];entries=json.loads((R/'run_matrix.json').read_text());rows={};arrays={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
with open('/data01/lyw/GLoMo/datasets/mosei.pkl','rb') as f:examples=pickle.load(f)['dev']
yref=np.array([np.asarray(x[1]).reshape(-1)[0] for x in examples]);arrays['sample_ids']=np.array([str(x[2].decode() if isinstance(x[2],bytes) else x[2]) for x in examples]);arrays['labels']=yref
for e in entries:
 D=Path(e['root']);p=json.loads((D/'protocol/protocol.json').read_text());pair={}
 for v in ['dynamic','constant']:
  d=D/'runs'/('{}_mosei_{}_seed5576'.format(p['version'],v));m=json.loads((d/'metrics.json').read_text());hist=[json.loads(x) for x in (d/'history.jsonl').read_text().splitlines()];assert len(hist)==100 and m['test_evaluated'] is False and m['evaluation_split']=='dev'
  assert m['best_epoch']==min(range(100),key=lambda i:hist[i]['valid_mae']);assert sha(d/'best_model.pt')==m['checkpoint_sha256']
  z=np.load(d/'dev_predictions.npz');y=z['labels'];pred=z['predictions'];assert pred.shape==y.shape and np.isfinite(pred).all() and np.allclose(y,yref,atol=1e-6);arrays[e['key']+'_'+v]=pred
  nz=y!=0;yt=y>=0;yp=pred>=0;yn=y[nz]>0;pn=pred[nz]>0
  met=dict(mae=float(np.abs(pred-y).mean()),mse=float(np.mean((pred-y)**2)),corr=float(np.corrcoef(pred,y)[0,1]),acc7=float(np.mean(np.round(np.clip(pred,-3,3))==np.round(np.clip(y,-3,3)))),acc2=float(np.mean(yp==yt)),f1=float(f1_score(yt,yp,average='weighted')),acc2_non_zero=float(np.mean(pn==yn)),f1_non_zero=float(f1_score(yn,pn,average='weighted')))
  assert abs(met['mae']-m['valid_mae'])<1e-6
  pair[v]=dict(metrics=met,selected_epoch_one_based=m['best_epoch']+1,checkpoint_sha256=m['checkpoint_sha256'])
 pair['delta_dynamic_minus_constant']={k:pair['dynamic']['metrics'][k]-pair['constant']['metrics'][k] for k in met};rows[e['key']]=pair
factor_effects={}
for v in ['dynamic','constant']:
 factor_effects[v]={
 'readout_change_at_sentence_objective':rows['sentence_word'][v]['metrics']['mae']-rows['sentence_sentence'][v]['metrics']['mae'],
 'readout_change_at_joint_objective':rows['joint_word'][v]['metrics']['mae']-rows['joint_sentence'][v]['metrics']['mae'],
 'objective_change_at_sentence_readout':rows['joint_sentence'][v]['metrics']['mae']-rows['sentence_sentence'][v]['metrics']['mae'],
 'objective_change_at_word_readout':rows['joint_word'][v]['metrics']['mae']-rows['sentence_word'][v]['metrics']['mae']}
report=dict(dataset='mosei',split='dev',pretraining_seed=5576,downstream_seed=5576,per_configuration=rows,factor_effects_mae=factor_effects,lowest_dynamic_dev_mae_configuration=min(rows,key=lambda k:rows[k]['dynamic']['metrics']['mae']),scope='Single-seed candidate screening; all cells use original fixed100 epochs and earliest min dev MAE; F1 true-label support; zero nonnegative for Acc2, excluded for nonzero; no test evaluation; candidate confirmation uses additional seeds.')
(R/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(R/'dev_predictions.npz',**arrays);print(json.dumps(report,indent=2))
