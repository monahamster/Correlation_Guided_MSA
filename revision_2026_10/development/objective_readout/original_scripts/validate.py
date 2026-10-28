from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];D=R/'mosei_seed5576_word_sentence';B=Path('/data01/lyw/GLoMo_length_pretrain_20261020')
sys.path.insert(0,str(D/'scripts'));import pilot as p
torch.set_num_threads(4)
for e in json.loads((R/'run_matrix.json').read_text()):
 d=Path(e['root'])
 if not e['reused']:
  for k,v in json.loads((d/'frozen_source.json').read_text()).items():assert p.sha(d/k)==v
  assert (d/'scripts/correlation_models.py').read_bytes()==(B/'scripts/correlation_models.py').read_bytes()
p.seed(5576);m=p.CorrelationModel(text_in_dim=768,vision_in_dim=35).cuda();ref=json.loads((B/'runs/mosei_same_length/inputs.json').read_text());assert p.state_sha(m)==ref['initial_state_sha256']
train,tm=p.load('mosei','train');dev,dm=p.load('mosei','dev');assert tm==ref['train'] and dm==ref['dev']
lengths=np.array([x['length'] for x in train]);d=p.donors(lengths,'same_length',105577);idx=np.arange(8);b=p.batch(train,idx,78,'cuda');nb=p.batch(train,d[idx],78,'cuda');assert torch.equal(b[1],nb[1])
x=torch.randn(2,4,3,64,device='cuda',requires_grad=True);y=torch.randn_like(x);mask=torch.tensor([[1,1,0,0],[1,1,1,0]],dtype=torch.bool,device='cuda')
actual=p.word_loss_rows(x,y,mask);expected=torch.stack([p.loss_rows(x[i,mask[i]],y[i,mask[i]]).mean() for i in range(2)]);assert torch.allclose(actual,expected,atol=1e-6);actual.sum().backward();assert torch.count_nonzero(x.grad[~mask])==0
m.eval()
with torch.no_grad():
 z=p.token_encode(m,b);original=torch.stack([(z[:,:,i]*b[1][:,:,None]).sum(1)/b[1].sum(1)[:,None] for i in range(3)],1);assert torch.allclose(p.pool_tokens(z,b[1]),original,atol=1e-6)
m.train();pt=p.token_encode(m,b);nt=p.token_encode(m,nb);loss=(.5*p.loss_rows(p.pool_tokens(pt,b[1]),p.pool_tokens(nt,nb[1]))+.5*p.word_loss_rows(pt,nt,b[1])).mean();assert torch.isfinite(loss)
opt=torch.optim.Adam(m.parameters(),lr=p.P['learning_rate']);before=p.state_sha(m);opt.zero_grad();loss.backward();assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None);opt.step();assert before!=p.state_sha(m)
for seed in [5576,42,2026]:
 for epoch in [1,20]:
  a=p.donors(lengths,'same_length',seed+100000+epoch);assert np.array_equal(lengths,lengths[a])
report={'passed':True,'model_source_identical_to_mosei_reference':True,'seed5576_initial_state_matches_reference':True,'train_dev_metadata_equal':True,'word_loss_manual_and_padding_gradient_checks':True,'sentence_pooling_equivalent':True,'finite_optimizer_update':True,'train_eligible':len(train),'dev_eligible':len(dev)}
p.save_json(R/'validation.json',report);print(json.dumps(report,indent=2))
