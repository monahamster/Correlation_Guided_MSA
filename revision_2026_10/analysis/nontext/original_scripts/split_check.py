from pathlib import Path
import numpy as np,json,datetime
from sklearn.model_selection import KFold,GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
R=Path(__file__).resolve().parents[1]
z=np.load('/data01/lyw/GLoMo_mosi_data_baselines_20261022/features.npz')
y=z['train_y'];g=z['train_groups'];seed=5576
protocol=dict(scope='Training-only descriptive split-sensitivity diagnostic; no tuning, no test evaluation',splits={'random':'KFold(5,shuffle=True,random_state=5576), video overlap allowed','group':'GroupKFold(5), no video overlap'},models={'ridge':'StandardScaler fit inside fold; Ridge(alpha=1000,solver=svd)','extra_trees':'200 trees, max_depth=8, min_samples_leaf=5, max_features=1.0, seed=5576, n_jobs=4'},features='Existing pooled GLoMo features: thirds means and q10/q50/q90, same as prior simple baselines',interpretation='Different fold constructions confound video exclusion and fold composition; a gap is descriptive, not causal proof. Hyperparameters retained from prior diagnostic; Ridge alpha was previously chosen on training group CV, so these are descriptive scores, not an independent nested-CV estimate.',test_evaluated=False)
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
report=dict(protocol=protocol,results={},complete=False);saved={}
def met(p):
 return dict(mae=float(np.mean(np.abs(p-y))),corr=float(np.corrcoef(p,y)[0,1]) if p.std()>1e-10 else None,pred_std=float(p.std()))
for split,cv in [('random',KFold(5,shuffle=True,random_state=seed)),('group',GroupKFold(5))]:
 folds=list(cv.split(y,y,g));constant=np.empty_like(y)
 for tr,va in folds:constant[va]=np.mean(y[tr])
 report['results'][split]=dict(constant=met(constant),folds=[dict(train_samples=len(tr),valid_samples=len(va),overlapping_videos=len(set(g[tr])&set(g[va]))) for tr,va in folds],models={})
 for variant in ['a','v','av']:
  x=z['train_'+variant]
  for name in ['ridge','extra_trees']:
   pred=np.empty_like(y)
   for i,(tr,va) in enumerate(folds):
    if split=='group':assert not(set(g[tr])&set(g[va]))
    model=make_pipeline(StandardScaler(),Ridge(alpha=1000,solver='svd')) if name=='ridge' else ExtraTreesRegressor(n_estimators=200,max_depth=8,min_samples_leaf=5,max_features=1.0,random_state=seed,n_jobs=4)
    model.fit(x[tr],y[tr]);pred[va]=model.predict(x[va])
   result=met(pred);result['gain_vs_fold_train_mean']=report['results'][split]['constant']['mae']-result['mae']
   report['results'][split]['models'][variant+'_'+name]=result;saved[split+'_'+variant+'_'+name]=pred
   (R/'cv_results.json').write_text(json.dumps(report,indent=2)+'\n')
   print(split,variant,name,json.dumps(result),flush=True)
report['complete']=True;report['finished_utc']=datetime.datetime.utcnow().isoformat()+'Z'
(R/'cv_results.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(R/'cv_predictions.npz',labels=y,groups=g,**saved)
print('CV_COMPLETE',flush=True)
