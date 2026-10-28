from pathlib import Path
import json,datetime,collections,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'evaluation_manifest.json').read_text())
variants={x['id']:x['variant'] for x in manifest['table']}
groups=collections.defaultdict(list);digests=collections.defaultdict(set);models=[]
def flatten(d,p=''):
    out={}
    for k,v in d.items():
        key=(p+'.'+k).strip('.')
        if isinstance(v,dict):out.update(flatten(v,key))
        elif isinstance(v,(int,float)) and not isinstance(v,bool):out[key]=float(v)
    return out
for e in manifest['entries']:
    p=R/'evaluations'/e['id']/'progress.json';v=json.loads(p.read_text())
    assert v['status']=='complete' and v['completed']==v['total']==19
    assert v['checkpoint_sha256']==e['checkpoint_sha256']
    assert v['protocol_sha256']==hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()
    models.append(dict(id=e['id'],clean_replay_max_abs_diff=v['clean_replay_max_abs_diff'],gate_error=v['gate_reconstruction_max_abs_diff']))
    perseed=collections.defaultdict(list)
    for row in v['rows']:
        key=(e['dataset'],row['condition'])
        digests[key].add(row['input_audit']['input_digest'])
        arr=np.load(p.parent/(row['condition']+'.npz'))
        assert len(arr['predictions'])==e['samples']['test']
        assert np.isfinite(arr['predictions']).all()
        for mod in ['t','a','v']:
            g=arr[mod+'_gates']
            assert np.allclose(g.sum(1),1) and np.all((g>0).sum(1)==2)
        data=flatten(row['metrics'],'performance');data.update(flatten(row['routing'],'routing'))
        data['mae_degradation']=row['mae_degradation']
        perseed[(row['kind'],row['level'])].append(data)
    for (kind,level),rows in perseed.items():
        assert len(rows)==(1 if kind=='clean' else 3)
        vals={k:float(np.mean([row[k] for row in rows])) for k in rows[0]}
        groups[(e['dataset'],variants[e['id']],kind,level)].append(dict(seed=e['seed'],values=vals))
assert all(len(v)==1 for v in digests.values()),'Unpaired corruption inputs'
summary=[]
for (ds,variant,kind,level),rows in sorted(groups.items()):
    assert len(rows)==5
    keys=rows[0]['values'].keys()
    agg={k:dict(mean=float(np.mean([r['values'][k] for r in rows])),sd=float(np.std([r['values'][k] for r in rows],ddof=1))) for k in keys}
    summary.append(dict(dataset=ds,variant=variant,kind=kind,level=level,n_training_seeds=5,corruption_replicates=1 if kind=='clean' else 3,statistics=agg,per_training_seed=rows))
# Paired degradation contrasts at each severity and normalized curve integral.
contrasts=[]
for ds in ['mosi','mosei']:
    for kind in ['audio','visual']:
        levels=[.1,.3,.5] if kind=='audio' else [.2,.4,.6]
        for control in ['baseline','constant']:
            curves=[]
            for level in levels:
                d={r['seed']:r['values'] for r in groups[(ds,'dynamic',kind,level)]}
                c={r['seed']:r['values'] for r in groups[(ds,control,kind,level)]}
                vals=[d[s]['mae_degradation']-c[s]['mae_degradation'] for s in sorted(d)]
                contrasts.append(dict(dataset=ds,kind=kind,level=level,contrast='dynamic-minus-'+control,metric='paired_MAE_degradation',mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),seeds=sorted(d),values=vals))
                curves.append(vals)
            area=np.trapz(np.asarray([[0.]*5]+curves),x=[0.]+levels,axis=0)/levels[-1]
            contrasts.append(dict(dataset=ds,kind=kind,contrast='dynamic-minus-'+control,metric='normalized_area_of_MAE_degradation',mean=float(area.mean()),sd=float(area.std(ddof=1)),values=area.tolist()))
out=dict(created_utc=datetime.datetime.utcnow().isoformat()+'Z',models=30,dataset_passes=570,input_pairing_verified=True,all_predictions_and_gates_verified=True,clean_replay= models,summary=summary,paired_degradation_contrasts=contrasts,interpretation='Corruption replicates averaged within training seed; five trained seeds are statistical units. Routing counterfactual changes logits only and is not an outcome intervention.')
(R/'comparison.json').write_text(json.dumps(out,indent=2,allow_nan=False))
lines=['Progressive feature corruption and routing analysis completed: 30 checkpoints, 570 conditions.','Mean +/- sample SD across five training seeds; three corruptions averaged within seed.']
for row in summary:
    s=row['statistics'];lines.append('%s %s %s %.1f: MAE %.6f +/- %.6f; MAE degradation %+.6f'%(row['dataset'],row['variant'],row['kind'],row['level'],s['performance.mae']['mean'],s['performance.mae']['sd'],s['mae_degradation']['mean']))
(R/'results.txt').write_text('\n'.join(lines)+'\n')
print('SUMMARY_COMPLETE',len(summary),flush=True)
