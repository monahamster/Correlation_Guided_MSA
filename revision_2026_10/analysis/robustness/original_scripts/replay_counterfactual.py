from pathlib import Path
import os,json,shutil,datetime
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import torch
torch.set_num_threads(4)
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'evaluation_manifest.json').read_text())
changed=[];max_error=0.;old_constant_max=0.;new_constant_max=0.
def gates(logits,batch_size):
    out=[]
    for start in range(0,len(logits),batch_size):
        x=torch.from_numpy(logits[start:start+batch_size].copy()).cuda()
        top,idx=x.topk(3,dim=1)
        g=torch.zeros_like(x).scatter(1,idx[:,:2],torch.softmax(top[:,:2],dim=1))
        out.append(g.cpu().numpy())
    return np.concatenate(out)
for e in manifest['entries']:
    d=R/'evaluations'/e['id'];p=d/'progress.json';v=json.loads(p.read_text())
    shutil.copy2(p,d/'progress_initial_numpy_counterfactual.json')
    clean=np.load(d/'clean_0.0_seed0.npz')
    batch_size=json.loads((Path(e['run'])/'run_config.json').read_text())['arguments']['test_batch_size']
    for row in v['rows']:
        arr=np.load(d/(row['condition']+'.npz'))
        for mod in ['t','a','v']:
            g=arr[mod+'_gates'];base=arr[mod+'_base_logits'];bias=arr[mod+'_rel_bias']
            expected=gates(base+bias,batch_size);err=float(np.max(np.abs(g-expected)));max_error=max(max_error,err)
            assert err<1e-6 and np.array_equal(g>0,expected>0),(e['id'],row['condition'],mod,err)
            cf=gates(base+clean[mod+'_rel_bias'],batch_size)
            rr=row['routing'][mod]
            old=rr['fixed_feature_reliability_top2_switch']
            new=float(((g>0)!=(cf>0)).any(1).mean())
            new_l1=float(np.abs(g-cf).sum(1).mean())
            if old!=new:changed.append(dict(id=e['id'],condition=row['condition'],modality=mod,old_top2=old,new_top2=new))
            if '_constant_' in e['id'] or '_baseline_' in e['id']:
                old_constant_max=max(old_constant_max,old);new_constant_max=max(new_constant_max,new)
                assert new==0. and new_l1<1e-6
            rr['fixed_feature_reliability_gate_l1']=new_l1
            rr['fixed_feature_reliability_top2_switch']=new
            l=base+clean[mod+'_rel_bias'];sort=np.sort(l,axis=1)
            rr['counterfactual_topk_boundary_tie_fraction']=float((sort[:,0]==sort[:,1]).mean())
    v['counterfactual_postprocess']='PyTorch CUDA topk(k+1) and softmax, original inference batch size; actual gates and supports replay verified.'
    p.write_text(json.dumps(v,indent=2,allow_nan=False))
for name in ['comparison.json','validation.json','results.txt','completion_manifest.json']:
    p=R/name
    if p.exists():shutil.copy2(p,R/('initial_'+name))
out=dict(completed_utc=datetime.datetime.utcnow().isoformat()+'Z',max_actual_gate_replay_error=max_error,changed_top2_statistics=changed,old_control_max_switch=old_constant_max,new_control_max_switch=new_constant_max,reason='Use original CUDA topk tie handling for fixed-feature counterfactuals; original evaluation predictions and actual gates unchanged.')
(R/'counterfactual_replay_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
