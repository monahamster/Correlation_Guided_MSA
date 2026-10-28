from pathlib import Path
import json,datetime,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
B=Path('/data01/lyw/GLoMo_mosi_data_baselines_20261022')
D=Path('/data01/lyw/GLoMo_modality_ablation_20261021/diagnostics_mosi')
z=np.load(B/'features.npz'); y=z['dev_y']; groups=z['dev_groups']; vids=np.unique(groups)
def metrics(p,t):
 return dict(mae=float(np.abs(p-t).mean()),corr=float(np.corrcoef(p,t)[0,1]) if np.std(p)>1e-10 and np.std(t)>1e-10 else None,pred_std=float(np.std(p)),label_std=float(np.std(t)),bias=float(np.mean(p-t)))
constant=np.full_like(y,z['train_y'].mean()); base=np.abs(constant-y)
out=dict(created_utc=datetime.datetime.utcnow().isoformat()+'Z',test_evaluated=False,video_count=len(vids),constant=metrics(constant,y),runs={},feature_shift={},labels={})
for split in ['train','dev']:
 t=z[split+'_y'];g=z[split+'_groups'];means=np.array([t[g==v].mean() for v in g])
 out['labels'][split]=dict(samples=len(t),videos=len(np.unique(g)),mean=float(t.mean()),std=float(t.std()),positive_fraction=float((t>=0).mean()),between_video_variance_fraction=float(np.mean((means-t.mean())**2)/np.var(t)))
for v in ['t','a','v','av']:
 rows=[]
 for seed in [5576,42,2026]:
  pz=np.load(D/(v+'_seed'+str(seed)+'_dev.npz'));assert np.array_equal(pz['labels'],y)
  p=pz['predictions'];err=np.abs(p-y)
  rows.append(dict(seed=seed,overall=metrics(p,y),mae_gain_vs_constant=float((base-err).mean()),videos=[dict(video=str(g),samples=int((groups==g).sum()),label_mean=float(y[groups==g].mean()),**metrics(p[groups==g],y[groups==g]),gain_vs_constant=float((base[groups==g]-err[groups==g]).mean())) for g in vids],leave_one_video_out_gain=[float((base[groups!=g]-err[groups!=g]).mean()) for g in vids]))
 out['runs'][v]=rows
 out.setdefault('summary',{})[v]=dict(mean_mae=float(np.mean([x['overall']['mae'] for x in rows])),mean_corr=float(np.mean([x['overall']['corr'] for x in rows])),mean_pred_std=float(np.mean([x['overall']['pred_std'] for x in rows])),mean_gain_vs_constant=float(np.mean([x['mae_gain_vs_constant'] for x in rows])),positive_video_seed_pairs=sum(x['gain_vs_constant']>0 for row in rows for x in row['videos']),total_video_seed_pairs=len(rows)*len(vids),leave_one_video_out_gain_range=[float(np.min([x for row in rows for x in row['leave_one_video_out_gain']])),float(np.max([x for row in rows for x in row['leave_one_video_out_gain']]))])
for v in ['a','v','av']:
 x=z['train_'+v];d=z['dev_'+v];sd=x.std(0);valid=sd>1e-6
 effect=np.abs(d.mean(0)[valid]-x.mean(0)[valid])/sd[valid]
 ratio=d.std(0)[valid]/sd[valid]
 out['feature_shift'][v]=dict(dimensions=x.shape[1],near_constant_train_dimensions=int((~valid).sum()),standardized_mean_difference_quantiles=np.quantile(effect,[.5,.9,.99,1]).tolist(),std_ratio_quantiles=np.quantile(ratio,[0,.5,.9,1]).tolist(),near_constant_columns_max_abs_dev=float(np.abs(d[:,~valid]).max()) if (~valid).any() else None,near_constant_columns_dev_std_max=float(d[:,~valid].std(0).max()) if (~valid).any() else None)
manifest={}
for p in [B/'features.npz',B/'audit.json',B/'baseline_results.json',D/'report.json',Path('/data01/lyw/GLoMo_mosi_nontext_fit_20261021/comparison.json')]:
 manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['labels','summary','feature_shift']},indent=2),flush=True)
