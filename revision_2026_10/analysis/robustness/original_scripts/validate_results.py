from pathlib import Path
import json,hashlib,collections,datetime
import numpy as np
from metrics import sentiment_metrics
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'evaluation_manifest.json').read_text())
variants={e['id']:e['variant'] for e in manifest['table']}
reference={};max_q_diff=0.;max_metric_diff=0.;count=0;hashes={}
for e in manifest['entries']:
    v=json.loads((R/'evaluations'/e['id']/'progress.json').read_text())
    clean=np.load(R/'evaluations'/e['id']/'clean_0.0_seed0.npz')
    for row in v['rows']:
        p=R/'evaluations'/e['id']/(row['condition']+'.npz')
        arr=np.load(p)
        assert np.array_equal(arr['labels'],clean['labels'])
        assert np.array_equal(arr['sample_ids'],clean['sample_ids'])
        m=sentiment_metrics(arr['predictions'],arr['labels'])
        for key,val in m.items():
            if isinstance(val,(int,float)):
                err=abs(val-row['metrics'][key]);max_metric_diff=max(max_metric_diff,err);assert err<1e-10,(e['id'],key,err)
        if variants[e['id']]!='baseline':
            key=(e['dataset'],row['condition'])
            q=np.stack([arr['q_'+mod] for mod in ['t','a','v']],axis=1)
            if key in reference:
                err=float(np.max(np.abs(q-reference[key])));max_q_diff=max(max_q_diff,err);assert err<2e-5,(e['id'],row['condition'],err)
            else:reference[key]=q.copy()
            for mod in ['t','a','v']:
                if variants[e['id']]=='dynamic':
                    assert np.allclose(arr['s_'+mod],1+.2*(arr['q_'+mod]-.5),atol=1e-7)
                else:
                    assert np.array_equal(arr['s_'+mod],clean['s_'+mod])
                    assert np.array_equal(arr[mod+'_rel_bias'],clean[mod+'_rel_bias'])
        hashes[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();count+=1
assert count==570
for path,digest in json.loads((R/'execution_manifest.json').read_text()).items():
    assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
result=dict(passed=True,conditions=570,metrics_recomputed=True,max_metric_diff=max_metric_diff,paired_labels_and_ids=True,frozen_correlation_scores_match_across_guided_models=True,max_q_diff=max_q_diff,constant_scales_and_bias_invariant=True,dynamic_scales_match_formula=True,evaluation_sources_unchanged=True,created_utc=datetime.datetime.utcnow().isoformat()+'Z')
(R/'validation.json').write_text(json.dumps(result,indent=2));(R/'predictions_manifest.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(result),flush=True)
