from pathlib import Path
import sys,json,hashlib,copy
import torch,numpy as np
R=Path(__file__).resolve().parent;D=R/'word_readout_candidate';sys.path.insert(0,str(D))
from readout import load_model,score,score_encoded
sys.path.insert(0,'/data01/lyw/GLoMo_mosi_pretrain_opt_20261020/word_sentence_triplet/scripts')
import pilot as p
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.enabled=False
model=load_model();records,meta=p.load('mosi','dev');train,tm=p.load('mosi','train');train_all=torch.load(tm['path'],map_location='cpu')['records']
@torch.no_grad()
def run(rows):
 out={k:[] for k in ['cosines','q','scales']}
 for start in range(0,len(rows),32):
  x,m=p.batch(rows,np.arange(start,min(start+32,len(rows))),58,'cuda');a=score(model,x['text'],x['audio'],x['visual'],m)
  for k in out:out[k].append(a[k].cpu().numpy())
 return {k:np.concatenate(v) for k,v in out.items()}
clean=run(records);train_out=run(train_all)
assert all(np.isfinite(x).all() for x in clean.values());assert np.all((clean['q']>=0)&(clean['q']<=1));assert np.all((clean['scales']>=.9)&(clean['scales']<=1.1))
x,m=p.batch(records,np.arange(4),58,'cuda')
with torch.no_grad():
 z=model(x['text'],x['audio'],x['visual'],m,m,m);out=score_encoded(z,m)
 manual=[]
 for i in range(4):
  manual.append(torch.stack([torch.nn.functional.cosine_similarity(z[a][i,m[i]],z[b][i,m[i]],dim=-1).mean() for a,b in [(0,1),(0,2),(1,2)]]))
 assert torch.allclose(out['cosines'],torch.stack(manual),atol=1e-6)
 changed=[a.clone() for a in z]
 for a in changed:a[~m]=1234
 assert torch.allclose(score_encoded(changed,m)['q'],out['q'],atol=1e-6)
 assert torch.allclose(score_encoded([a.flip(1) for a in z],m.flip(1))['q'],out['q'],atol=1e-6)
 try:score_encoded(z,torch.zeros_like(m))
 except ValueError:pass
 else:raise AssertionError('Empty mask accepted')
checkpoint_before=p.state_sha(model);conditions=json.loads(Path('/data01/lyw/GLoMo_diagnostics_20261020/diagnostic_protocol.json').read_text())['conditions'];response={};arrays={}
for condition in conditions:
 g=torch.Generator().manual_seed(20261020);key='visual' if condition['modality']=='vision' else 'audio';index=2 if key=='visual' else 1;altered=[]
 for r in records:
  rr=dict(r);data=r[key].clone();n=r['length'];level=condition['level']
  if condition['kind']=='noise':data+=level*data.std().clamp_min(1e-6)*torch.randn(data.shape,generator=g)
  else:data[torch.rand(n,generator=g)<level]=0
  rr[key]=data;altered.append(rr)
 v=run(altered);delta=v['scales'][:,index]-clean['scales'][:,index]
 response[condition['name']]=dict(modality=key,scale_delta_mean=float(delta.mean()),fraction_decreased=float((delta<0).mean()),q_mean=float(v['q'][:,index].mean()));arrays[condition['name']]=v['scales']
zero=[{**r,**{k:torch.zeros_like(r[k]) for k in p.MODS}} for r in records];null=run(zero)
assert p.state_sha(model)==checkpoint_before
summary=dict(passed=True,scope='Standalone frozen scorer validation plus branch-only response; no downstream model or test evaluation.',manual_aligned_cosine=True,padding_ignored=True,joint_temporal_permutation_invariant=True,empty_masks_rejected=True,model_unchanged=True,eligible_dev_samples=len(records),complete_train_samples=len(train_all),train_scale_means=train_out['scales'].mean(0).tolist(),dev_clean_q_mean=clean['q'].mean(0).tolist(),dev_all_zero_q_mean=null['q'].mean(0).tolist(),corruption_response=response)
p.save_json(D/'validation.json',summary);np.savez_compressed(D/'signals.npz',**{'clean_'+k:v for k,v in clean.items()},train_scale_means=train_out['scales'].mean(0),all_zero_q=null['q'],**arrays)
p.save_json(D/'frozen_source.json',{f.name:p.sha(f) for f in [D/'protocol.json',D/'correlation_models.py',D/'readout.py']})
print(json.dumps(summary,indent=2),flush=True)
