from pathlib import Path
import json,hashlib,itertools,datetime
import numpy as np
from scipy import stats
from metrics import sentiment_metrics
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'evaluation_manifest.json').read_text());jobs=json.loads((R/'test_jobs.json').read_text())
assert len(jobs)==78 and all(j['state']=='complete' for j in jobs.values())
results={};reference={}
for e in manifest['entries']:
 d=R/'evaluations/test'/e['id'];m=json.loads((d/'metrics.json').read_text())
 assert hashlib.sha256((d/'predictions.npz').read_bytes()).hexdigest()==jobs[e['id']]['predictions_sha256']
 z=np.load(d/'predictions.npz');pred=z['predictions'];y=z['labels'];ids=z['sample_ids']
 assert len(ids)==len(set(ids))==e['samples']['test']
 if e['dataset'] not in reference:reference[e['dataset']]=(y.copy(),ids.copy())
 ry,ri=reference[e['dataset']];assert np.array_equal(y,ry) and np.array_equal(ids,ri)
 met=sentiment_metrics(pred,y);assert met==m['metrics']
 assert m['checkpoint_sha256']==e['checkpoint_sha256'] and m['config_sha256']==e['config_sha256'] and m['split']=='test' and m['limit'] is None
 results[e['id']]=dict(metrics=met,checkpoint=e['checkpoint'],checkpoint_sha256=e['checkpoint_sha256'],selected_epoch_one_based=e['best_epoch_zero_based']+1)
names=['mae','mse','corr','acc7','acc2','f1','acc2_non_zero','f1_non_zero'];tables={}
for table,members in manifest['tables'].items():
 tables[table]={}
 for ds in ['mosi','mosei']:
  table_rows={}
  variants=list(dict.fromkeys(x['variant'] for x in members if x['dataset']==ds))
  for variant in variants:
   group=sorted([x for x in members if x['dataset']==ds and x['variant']==variant],key=lambda x:x['seed'])
   expected={5576,42,2026,2027,3407} if table=='main' else {5576,42,2026}
   assert {x['seed'] for x in group}==expected
   table_rows[variant]=dict(seeds=[x['seed'] for x in group],per_seed={str(x['seed']):results[x['id']]['metrics'] for x in group},summary={k:dict(mean=float(np.mean([results[x['id']]['metrics'][k] for x in group])) if all(results[x['id']]['metrics'][k] is not None for x in group) else None,sample_sd=float(np.std([results[x['id']]['metrics'][k] for x in group],ddof=1)) if all(results[x['id']]['metrics'][k] is not None for x in group) else None) for k in names})
  tables[table][ds]=table_rows
pairs=[]
for ds in ['mosi','mosei']:
 rows=tables['main'][ds];seeds=[5576,42,2026,2027,3407]
 for control in ['baseline','constant']:
  differences=np.array([rows['dynamic']['per_seed'][str(s)]['mae']-rows[control]['per_seed'][str(s)]['mae'] for s in seeds])
  mean=float(differences.mean());sd=float(differences.std(ddof=1));se=sd/np.sqrt(len(seeds))
  statistic=float(mean/se) if se else None
  p=float(2*stats.t.sf(abs(statistic),df=4)) if statistic is not None else (1.0 if mean==0 else 0.0)
  critical=float(stats.t.ppf(.975,df=4));permuted=np.array([np.mean(differences*np.array(sign)) for sign in itertools.product([-1,1],repeat=5)])
  pairs.append(dict(dataset=ds,comparison='dynamic_minus_'+control,seeds=seeds,per_seed_difference=differences.tolist(),mean_difference=mean,sample_sd=sd,mean_difference_95pct_t_interval=[mean-critical*se,mean+critical*se],paired_t_statistic=statistic,paired_t_p_two_sided=p,exact_signflip_p_two_sided=float(np.mean(np.abs(permuted)>=abs(mean)-1e-15)),dynamic_better_count=int((differences<0).sum()),additional_two_mean_difference=float(differences[-2:].mean())))
order=np.argsort([x['paired_t_p_two_sided'] for x in pairs]);previous=0.
for rank,i in enumerate(order):
 value=min(1.,max(previous,(len(pairs)-rank)*pairs[i]['paired_t_p_two_sided']));pairs[i]['paired_t_p_holm']=value;previous=value
report=dict(state='complete',finished_utc=datetime.datetime.utcnow().isoformat()+'Z',scope='Clean held-out test evaluation. Main five downstream seeds; module/modality three seeds. Fixed pretrained seed5576; checkpoint selection exclusively by earlier dev MAE.',unique_checkpoints=len(results),test_samples={ds:len(v[0]) for ds,v in reference.items()},tables=tables,primary_mae_paired_tests=pairs,results=results,statistics='Primary two-sided paired t tests, Holm adjustment over four dataset/control comparisons; t intervals unadjusted. Exact paired sign-flip p also reported, five-seed resolution minimum .0625. No instance bootstrap substituted for training seeds.',test_evaluated=True)
with (R/'test_comparison.json').open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
lines=['Clean test evaluation: 78 selected checkpoints; all samples and metrics verified.']
for ds in ['mosi','mosei']:
 lines.append(ds+' main (five seeds), mean +/- sample SD:')
 for variant,row in tables['main'][ds].items():lines.append(variant+' '+', '.join(k+'='+format(row['summary'][k]['mean'],'.6f')+' +/- '+format(row['summary'][k]['sample_sd'],'.6f') for k in names))
lines.append('Paired MAE tests: '+json.dumps(pairs))
(R/'test_summary.txt').write_text('\n'.join(lines)+'\n')
(R/'test_results_manifest.json').write_text(json.dumps({e['id']:dict(metrics_sha256=hashlib.sha256((R/'evaluations/test'/e['id']/'metrics.json').read_bytes()).hexdigest(),predictions_sha256=jobs[e['id']]['predictions_sha256']) for e in manifest['entries']},indent=2)+'\n')
print('SUMMARY_VERIFIED',len(results),report['test_samples'],flush=True)
