from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1]
report=json.loads((R/'test_comparison.json').read_text());manifest=json.loads((R/'evaluation_manifest.json').read_text())
mapping={(x['dataset'],x['seed'],x['variant']):x['id'] for x in manifest['tables']['main']}
groups_report={};decomposition={};module={}
for ds in ['mosi','mosei']:
 groups_report[ds]={};decomposition[ds]={}
 for seed in [5576,42,2026,2027,3407]:
  predictions={}
  for v in ['baseline','constant','dynamic']:
   z=np.load(R/'evaluations/test'/mapping[(ds,seed,v)]/'predictions.npz');predictions[v]=z['predictions'];y=z['labels']
  masks={'negative':y<0,'zero':y==0,'positive':y>0}
  groups_report[ds][str(seed)]={v:{g:dict(n=int(mask.sum()),mae=float(np.abs(pred[mask]-y[mask]).mean()) if mask.any() else None,acc2=float(np.mean((pred[mask]>=0)==(y[mask]>=0))) if mask.any() else None) for g,mask in masks.items()} for v,pred in predictions.items()}
  decomposition[ds][str(seed)]={}
  for control in ['baseline','constant']:
   gains={g:float((np.abs(predictions[control][mask]-y[mask])-np.abs(predictions['dynamic'][mask]-y[mask])).sum()/len(y)) for g,mask in masks.items()}
   overall=float(np.mean(np.abs(predictions[control]-y)-np.abs(predictions['dynamic']-y)))
   assert abs(sum(gains.values())-overall)<1e-12
   decomposition[ds][str(seed)][control]=dict(weighted_mae_improvement_by_label_group=gains,total_mae_improvement=overall)
 module[ds]={}
 for seed in [5576,42,2026]:
  rows=report['tables']['module'][ds];v={name:r['per_seed'][str(seed)]['mae'] for name,r in rows.items()}
  module[ds][str(seed)]=dict(fusion_minus_baseline=v['fusion_only']-v['baseline'],moe_minus_baseline=v['moe_only']-v['baseline'],full_minus_baseline=v['full']-v['baseline'],add_moe_to_fusion=v['full']-v['fusion_only'],add_fusion_to_moe=v['full']-v['moe_only'],interaction=v['full']-v['fusion_only']-v['moe_only']+v['baseline'])
out=dict(scope='Descriptive label-group decomposition and paired module comparisons; no retraining or selection from test results.',label_groups=groups_report,main_mae_improvement_decomposition=decomposition,module_paired_mae_effects=module)
(R/'test_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('DESCRIPTIVE_ANALYSIS_COMPLETE')
