from pilot import *
import importlib.util
REF=Path('/data01/lyw/GLoMo_length_pretrain_20261020')
def main():
 torch.set_num_threads(4)
 spec=importlib.util.spec_from_file_location('reference_models',REF/'scripts/correlation_models.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 checks={}
 for ds in P['datasets']:
  train,tm=load(ds,'train');dev,dm=load(ds,'dev');ri=json.loads((REF/'runs'/(ds+'_same_length')/'inputs.json').read_text())
  assert tm==ri['train'] and dm==ri['dev']
  seed(P['seed']);new=CorrelationModel(text_in_dim=768,vision_in_dim=47 if ds=='mosi' else 35).cuda()
  assert state_sha(new)==ri['initial_state_sha256']
  seed(P['seed']);reference=old.CorrelationModel(text_in_dim=768,vision_in_dim=47 if ds=='mosi' else 35).cuda();assert state_sha(reference)==state_sha(new)
  x=torch.randn(4,17,128,device='cuda');new.pos_encoder.train();reference.pos_encoder.train()
  seed(991);actual=new.pos_encoder(x);state=torch.cuda.get_rng_state()
  seed(991);expected=reference.pos_encoder.dropout(x);assert torch.equal(actual,expected);assert torch.equal(state,torch.cuda.get_rng_state())
  new.eval();reference.eval();assert torch.equal(new.pos_encoder(x),x)
  assert torch.allclose(reference.pos_encoder(x),new.pos_encoder(x)+new.pos_encoder.pe[:,:17])
  lengths=np.array([v['length'] for v in train]);h=json.loads((REF/'runs'/(ds+'_same_length')/'history.json').read_text())
  for epoch in range(1,21):
   d=donors(lengths,'same_length',P['seed']+P['train_negative_seed_offset']+epoch)
   order=np.random.RandomState(P['seed']+P['anchor_order_seed_offset']+epoch).permutation(len(train))
   assert hashlib.sha256(d.tobytes()).hexdigest()==h[epoch-1]['train_donors_sha256']
   assert hashlib.sha256(order.tobytes()).hexdigest()==h[epoch-1]['anchor_order_sha256']
  dl=np.array([v['length'] for v in dev]);saved=np.load(REF/'runs'/(ds+'_same_length')/'dev_donors.npz')
  for arm in P['arms']:
   for k in range(5):assert np.array_equal(donors(dl,arm,P['dev_negative_seed']+k),saved[arm][k])
  idx=np.arange(8);b=batch(train,idx,P['padding'][ds],'cuda');d=donors(lengths,'same_length',123)
  new.train();opt=torch.optim.Adam(new.parameters(),lr=P['learning_rate']);loss=loss_rows(encode(new,b),encode(new,batch(train,d[idx],P['padding'][ds],'cuda'))).mean();assert torch.isfinite(loss);loss.backward();opt.step()
  assert state_sha(new)!=ri['initial_state_sha256']
  checks[ds]={'shared_initial_state_with_E2b':True,'eligible_data_identical':True,'all_20_anchor_orders_and_donor_maps_identical':True,'all_dev_maps_identical':True,'position_omitted_dropout_and_rng_preserved':True,'finite_optimizer_step':True,'initial_state_sha256':ri['initial_state_sha256']}
  del train,dev,new,reference;torch.cuda.empty_cache()
 save_json(ROOT/'validation.json',{'passed':True,'checks':checks});print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
