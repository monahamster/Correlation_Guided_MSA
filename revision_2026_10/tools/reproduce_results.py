#!/usr/bin/env python3
"""Recompute every released clean result from predictions; export tables and plots."""
from pathlib import Path
import argparse,csv,json,hashlib,sys,itertools,shutil,subprocess
import numpy as np
from scipy import stats

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'analysis/clean_original'))
from metrics import sentiment_metrics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--plots',action='store_true');a=p.parse_args()
 if a.output.exists():p.error('Use a new output directory; archived results are never overwritten')
 registry=json.loads((R/'registry.json').read_text());expected=json.loads((R/'results/clean/test_comparison.json').read_text())
 original_hashes=json.loads((R/'results/clean/test_results_manifest.json').read_text())
 values={};arrays={};shared={}
 for rec in registry['runs']:
  rid=rec['id'];folder=R/'runs'/rid;fn=folder/'test_predictions.npz'
  assert hashlib.sha256(fn.read_bytes()).hexdigest()==original_hashes[rid]['predictions_sha256'],rid
  with np.load(fn,allow_pickle=False) as z:
   y=z['labels'].copy(); pred=z['predictions'].copy(); ids=z['sample_ids'].copy()
  assert len(ids)==len(set(ids))==rec['samples']['test']
  ds=rec['dataset']
  if ds not in shared:shared[ds]=(y,ids)
  assert np.array_equal(shared[ds][0],y) and np.array_equal(shared[ds][1],ids)
  computed=sentiment_metrics(pred,y);archived=json.loads((folder/'test_metrics.json').read_text())
  for key,val in computed.items():
   reference=archived['metrics'][key]
   if isinstance(val,(float,int)) and val is not None:assert np.isclose(val,reference,rtol=0,atol=1e-12),(rid,key)
   else:assert val==reference,(rid,key)
  assert archived['checkpoint_sha256']==rec['checkpoint_sha256']
  hist=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
  assert len(hist)==registry['epochs'][ds]
  assert min(hist,key=lambda x:x['valid_mae'])['epoch_zero_based']==rec['best_epoch_zero_based']
  values[rid]=computed;arrays[rid]=(y,pred)
 keys=['mae','mse','corr','acc2','f1','acc2_non_zero','f1_non_zero','acc7']
 rows=[];tables={}
 for table,members in registry['tables'].items():
  tables[table]={}
  for ds in ['mosi','mosei']:
   tables[table][ds]={}
   for variant in dict.fromkeys(x['variant'] for x in members if x['dataset']==ds):
    group=[x for x in members if x['dataset']==ds and x['variant']==variant]
    assert {x['seed'] for x in group}==({42,2026,2027,3407,5576} if table=='main' else {42,2026,5576})
    summary={}
    for key in keys:
     v=[values[x['id']][key] for x in group]
     mean=float(np.mean(v));sd=float(np.std(v,ddof=1));ref=expected['tables'][table][ds][variant]['summary'][key]
     assert np.isclose(mean,ref['mean'],atol=1e-12,rtol=0) and np.isclose(sd,ref['sample_sd'],atol=1e-12,rtol=0)
     summary[key]={'mean':mean,'sample_sd':sd}
     rows.append({'table':table,'dataset':ds,'variant':variant,'metric':key,'n_seeds':len(v),'mean':mean,'sample_sd':sd})
    tables[table][ds][variant]=summary
 paired=[]
 for ds in ['mosi','mosei']:
  for control in ['baseline','constant']:
   dif=np.array([values[f'{ds}_dynamic_seed{s}']['mae']-values[f'{ds}_{control}_seed{s}']['mae'] for s in [5576,42,2026,2027,3407]])
   se=dif.std(ddof=1)/np.sqrt(5);mean=float(dif.mean());critical=stats.t.ppf(.975,4)
   paired.append({'dataset':ds,'comparison':'dynamic_minus_'+control,'mean_difference':mean,'ci95':[float(mean-critical*se),float(mean+critical*se)],'p':float(2*stats.t.sf(abs(mean/se),4))})
 previous=0
 for rank,i in enumerate(np.argsort([x['p'] for x in paired])):
  previous=min(1.,max(previous,(4-rank)*paired[i]['p']));paired[i]['p_holm']=previous
 for x,ref in zip(paired,expected['primary_mae_paired_tests']):
  assert np.isclose(x['mean_difference'],ref['mean_difference'],atol=1e-12,rtol=0)
  assert np.isclose(x['p'],ref['paired_t_p_two_sided'],atol=1e-12,rtol=0)
  assert np.allclose(x['ci95'],ref['mean_difference_95pct_t_interval'],atol=1e-12,rtol=0)
  assert np.isclose(x['p_holm'],ref['paired_t_p_holm'],atol=1e-12,rtol=0)
 a.output.mkdir(parents=True)
 with (a.output/'summary.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (a.output/'summary.json').write_text(json.dumps({'tables':tables,'paired_mae':paired},indent=2)+'\n')
 if a.plots:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from sklearn.metrics import confusion_matrix
  for kind in ['all_binary','nonzero_binary','seven_level']:
   fig,axs=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
   for row,ds in enumerate(['mosi','mosei']):
    for col,variant in enumerate(['baseline','dynamic']):
     cms=[]
     for seed in [5576,42,2026,2027,3407]:
      y,pred=arrays[f'{ds}_{variant}_seed{seed}']
      if kind=='nonzero_binary':mask=y!=0;y=y[mask];pred=pred[mask];truth=(y>0).astype(int);guess=(pred>0).astype(int);labels=[0,1]
      elif kind=='all_binary':truth=(y>=0).astype(int);guess=(pred>=0).astype(int);labels=[0,1]
      else:truth=np.round(np.clip(y,-3,3)).astype(int);guess=np.round(np.clip(pred,-3,3)).astype(int);labels=list(range(-3,4))
      cms.append(confusion_matrix(truth,guess,labels=labels))
     cm=np.mean(cms,axis=0);pct=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(1,keepdims=True)!=0)*100
     ax=axs[row,col];ax.imshow(pct,vmin=0,vmax=100,cmap='Blues');ax.set(title=ds.upper()+' / '+variant,xlabel='Prediction',ylabel='Label',xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels)
     for i in range(len(labels)):
      for j in range(len(labels)):ax.text(j,i,f'{pct[i,j]:.1f}%',ha='center',va='center',color='white' if pct[i,j]>50 else 'black',fontsize=8)
   fig.savefig(a.output/(kind+'.png'),dpi=180);plt.close(fig)
  # Original numeric plotting scripts are run on temporary copies of their inputs in the new output directory.
  for name in ['robustness','efficiency']:
   dest=a.output/name;dest.mkdir();(dest/'scripts').mkdir()
   shutil.copy2(R/'analysis'/name/'comparison.json',dest/'comparison.json')
   shutil.copy2(R/'analysis'/name/'original_scripts/plot_results.py',dest/'scripts/plot_results.py')
   if name=='efficiency':shutil.copytree(R/'analysis/efficiency/evaluations',dest/'evaluations')
   subprocess.run([sys.executable,str((dest/'scripts/plot_results.py').resolve())],check=True)
 report={'clean_runs_verified':len(values),'summary_rows':len(rows),'paired_tests':len(paired),'prediction_hashes_verified':True,'checkpoint_selection_verified':True,'all_values_match_archived_results':True,'new_training_started':False}
 (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
