from pathlib import Path
import sys,json,torch,numpy as np
R=Path(__file__).resolve().parent;D=R/'word_sentence_triplet';sys.path.insert(0,str(D/'scripts'))
import pilot as p
torch.set_num_threads(4);p.seed(p.P['seed']);m=p.CorrelationModel(text_in_dim=768,vision_in_dim=47).cuda();ref=Path(json.loads((D/'reference.json').read_text())['root'])
old=json.loads((ref/'runs/mosi_same_length/inputs.json').read_text());assert p.state_sha(m)==old['initial_state_sha256']
assert (D/'scripts/correlation_models.py').read_bytes()==(ref/'scripts/correlation_models.py').read_bytes()
r,meta=p.load('mosi','train');assert meta==old['train'];v,vm=p.load('mosi','dev');assert vm==old['dev']
x=torch.randn(2,4,3,64,device='cuda',requires_grad=True);y=torch.randn_like(x);mask=torch.tensor([[1,1,0,0],[1,1,1,0]],device='cuda',dtype=torch.bool)
actual=p.word_loss_rows(x,y,mask)
expected=torch.stack([p.loss_rows(x[i,mask[i]],y[i,mask[i]]).mean() for i in range(2)])
assert torch.allclose(actual,expected,atol=1e-6);actual.sum().backward();assert torch.count_nonzero(x.grad[~mask])==0
b=p.batch(r,np.arange(4),58,'cuda');m.eval()
with torch.no_grad():
 out=m(b[0]['text'],b[0]['audio'],b[0]['visual'],b[1],b[1],b[1]);expected=torch.stack([(a*b[1][:,:,None]).sum(1)/b[1].sum(1)[:,None] for a in out],1)
 assert torch.allclose(p.encode(m,b),expected,atol=1e-6)
lengths=np.array([a['length'] for a in r]);donor=p.donors(lengths,'same_length',p.P['seed']+p.P['train_negative_seed_offset']+1);nb=p.batch(r,donor[:4],58,'cuda');assert torch.equal(b[1],nb[1])
m.train();pt=p.token_encode(m,b);nt=p.token_encode(m,nb);loss=(.5*p.loss_rows(p.pool_tokens(pt,b[1]),p.pool_tokens(nt,nb[1]))+.5*p.word_loss_rows(pt,nt,b[1])).mean();assert torch.isfinite(loss)
opt=torch.optim.Adam(m.parameters(),lr=p.P['learning_rate']);before=p.state_sha(m);loss.backward();opt.step();assert before!=p.state_sha(m)
p.save_json(D/'validation.json',dict(passed=True,initialization_identical=True,data_identical=True,word_loss_matches_manual=True,padding_gradients_zero=True,inference_aggregation_unchanged=True,finite_optimizer_update=True))
p.save_json(D/'frozen_source.json',{str(f.relative_to(D)):p.sha(f) for f in [D/'protocol.json',*sorted((D/'scripts').glob('*.py'))]})
print('WORD_SENTENCE_VALIDATION_PASSED',flush=True)
p.run('mosi','same_length')
