from pathlib import Path
import os,sys,json,hashlib,importlib.util
import torch,numpy as np
ROOT=Path(__file__).resolve().parent
D=ROOT/sys.argv[1];sys.path.insert(0,str(D/'scripts'))
import pilot as p
ref=Path(json.loads((D/'reference.json').read_text())['root'])
def validate():
 torch.set_num_threads(4);p.seed(p.P['seed']);model=p.CorrelationModel(text_in_dim=768,vision_in_dim=47).cuda()
 expected=json.loads((ref/'runs/mosi_same_length/inputs.json').read_text())
 assert p.state_sha(model)==expected['initial_state_sha256']
 assert (D/'scripts/correlation_models.py').read_bytes()==(ref/'scripts/correlation_models.py').read_bytes()
 records,meta=p.load('mosi','train');assert meta==expected['train'];dev,dm=p.load('mosi','dev');assert dm==expected['dev']
 lengths=np.array([x['length'] for x in records]);dl=np.array([x['length'] for x in dev])
 h=json.loads((ref/'runs/mosi_same_length/history.json').read_text())
 for epoch in range(1,21):
  donors=p.donors(lengths,'same_length',p.P['seed']+p.P['train_negative_seed_offset']+epoch);order=np.random.RandomState(p.P['seed']+p.P['anchor_order_seed_offset']+epoch).permutation(len(records))
  assert hashlib.sha256(donors.tobytes()).hexdigest()==h[epoch-1]['train_donors_sha256']
  assert hashlib.sha256(order.tobytes()).hexdigest()==h[epoch-1]['anchor_order_sha256']
 maps=np.load(ref/'runs/mosi_same_length/dev_donors.npz')
 for arm in ['random','same_length']:
  for k in range(5):assert np.array_equal(p.donors(dl,arm,p.P['dev_negative_seed']+k),maps[arm][k])
 b=p.batch(records,np.arange(4),58,'cuda');model.eval()
 with torch.no_grad():
  emb=p.encode(model,b);x,m=b;outputs=model(x['text'],x['audio'],x['visual'],m,m,m)
  manual=torch.stack([torch.stack([v[j,m[j]].max(0).values for j in range(4)]) for v in outputs],1)
  assert torch.allclose(emb,manual,atol=1e-6)
  changed={k:v.clone() for k,v in x.items()}
  for v in changed.values():v[~m]=123.0
  assert torch.allclose(emb,p.encode(model,(changed,m)),atol=1e-5)
 opt=torch.optim.Adam(model.parameters(),lr=p.P['learning_rate']);model.train();before=p.state_sha(model)
 loss=p.loss_rows(p.encode(model,b),p.encode(model,p.batch(records,np.arange(4,8),58,'cuda'))).mean();loss.backward();opt.step()
 assert torch.isfinite(loss) and p.state_sha(model)!=before
 p.save_json(D/'validation.json',dict(passed=True,initialization_identical=True,all_anchor_orders_and_donors_identical=True,mask_max_matches_manual=True,padding_invariant=True,finite_optimizer_update=True,reference=str(ref)))
validate()
files={str(f.relative_to(D)):p.sha(f) for f in [D/'protocol.json',*sorted((D/'scripts').glob('*.py'))]}
p.save_json(D/'frozen_source.json',files)
print('VALIDATED_AND_FROZEN',D,flush=True)
p.run('mosi','same_length')
