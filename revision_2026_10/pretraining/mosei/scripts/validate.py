from pilot import *
def check():
 torch.set_num_threads(4);device='cuda:0';checks={}
 for ds in P['datasets']:
  train,tm=load(ds,'train');dev,dm=load(ds,'dev');lengths=np.array([r['length'] for r in train])
  for arm in P['arms']:
   for epoch in range(1,P['epochs']+1):donors(lengths,arm,P['seed']+P['train_negative_seed_offset']+epoch)
  dl=np.array([r['length'] for r in dev])
  for arm in P['arms']:
   for k in range(5):donors(dl,arm,P['dev_negative_seed']+k)
  seed(P['seed']);m=CorrelationModel(text_in_dim=768,vision_in_dim=47 if ds=='mosi' else 35).to(device);initial=state_sha(m)
  seed(P['seed']);m2=CorrelationModel(text_in_dim=768,vision_in_dim=47 if ds=='mosi' else 35).to(device);assert initial==state_sha(m2)
  idx=np.arange(min(8,len(train)));b=batch(train,idx,P['padding'][ds],device);m.eval()
  with torch.no_grad():e=encode(m,b)
  assert torch.isfinite(e).all();assert torch.allclose(loss_rows(e,e),torch.full((len(idx),),.4,device=device),atol=1e-6)
  m.train();opt=torch.optim.Adam(m.parameters(),lr=P['learning_rate']);d=donors(lengths,'same_length',123)
  loss=loss_rows(encode(m,b),encode(m,batch(train,d[idx],P['padding'][ds],device))).mean();loss.backward();opt.step()
  assert state_sha(m)!=initial and torch.isfinite(loss)
  assert all(x.shape[1]==P['padding'][ds] for x in b[0].values())
  assert set(tm['eligible_ids']).isdisjoint(dm['eligible_ids'])
  checks[ds]={'train_total':tm['total'],'train_eligible':tm['eligible'],'dev_total':dm['total'],'dev_eligible':dm['eligible'],'initial_state_sha256':initial,'negative_maps':'all epochs and dev maps valid nonself permutations; exact-length maps preserve length','optimizer_step':'passed','self_triplet_loss':.4,'cache_hashes':'match frozen formal inputs'}
  del m,m2,train,dev;torch.cuda.empty_cache()
 save_json(ROOT/'validation.json',{'passed':True,'checks':checks});print(json.dumps(checks,indent=2))
if __name__=='__main__':check()
