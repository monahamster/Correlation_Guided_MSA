from pathlib import Path
import sys,pickle,json,re,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parent
S=Path('/data01/lyw/GLoMo_position_ablation_20261020')
sys.path.insert(0,str(S/'scripts'))
from pilot import load,donors,P,sha
records,meta=load('mosi','train')
with open('/data01/lyw/GLoMo/datasets/mosi.pkl','rb') as f:raw=pickle.load(f)['train']
labels={str(x[2].decode() if isinstance(x[2],bytes) else x[2]):float(np.asarray(x[1]).reshape(-1)[0]) for x in raw}
y=np.array([labels[r['sample_id']] for r in records]);lengths=np.array([r['length'] for r in records])
video=np.array([re.fullmatch(r'(.+)\[(\d+)\]',r['sample_id'])[1] for r in records]);rows=[]
for arm in ['random','same_length']:
 ds=np.stack([donors(lengths,arm,P['seed']+P['train_negative_seed_offset']+epoch) for epoch in range(1,21)])
 delta=np.abs(y[None]-y[ds]);rows.append(dict(negative_sampling=arm,label_distance_mean=float(delta.mean()),fraction_distance_le_05=float((delta<=.5).mean()),fraction_same_nonnegative_sign=float(((y[None]>=0)==(y[ds]>=0)).mean()),same_video_fraction=float((video[None]==video[ds]).mean())))
features={}
for k in ['text','audio','visual']:
 x=np.concatenate([r[k].numpy() if hasattr(r[k],'numpy') else r[k] for r in records]);std=x.std(0);means=np.stack([np.asarray(r[k]).mean(0) for r in records]);overall=means.mean(0)
 between=sum(len(g)*(means[g].mean(0)-overall)**2 for g in [np.flatnonzero(video==v) for v in np.unique(video)]).sum()
 total=((means-overall)**2).sum()
 features[k]=dict(shape=list(x.shape),channel_std_quantiles=np.quantile(std,[0,.1,.5,.9,1]).tolist(),channel_mean_abs_max=float(np.abs(x.mean(0)).max()),sample_mean_variance_explained_by_video=float(between/total))
report=dict(scope='MOSI training split only. Label distances are diagnosis; no labels are used by training here. Source-video grouping is descriptive, not a causal attribution.',input=meta,pairs=rows,features=features)
(R/'training_pair_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'pairs':rows,'features':features},indent=2),flush=True)
