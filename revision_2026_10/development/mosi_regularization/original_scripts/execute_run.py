from pathlib import Path
import sys,json,hashlib
import torch,numpy as np
R=Path(__file__).resolve().parent;D=R/sys.argv[1];sys.path.insert(0,str(D/'scripts'))
import pilot as p
for k,h in json.loads((D/'frozen_source.json').read_text()).items():assert p.sha(D/k)==h
seed=p.P['seed'];S=Path('/data01/lyw/GLoMo_mosi_pretrain_opt_20261020');ref=S/'word_sentence_triplet' if seed==5576 else S/'replications'/('seed{}_word_sentence'.format(seed))
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p.seed(seed);model=p.CorrelationModel(text_in_dim=768,vision_in_dim=47,dropout=p.P['regularization']['dropout']).cuda()
old=json.loads((ref/'runs/mosi_same_length/inputs.json').read_text());assert p.state_sha(model)==old['initial_state_sha256']
assert (D/'scripts/correlation_models.py').read_bytes()==(ref/'scripts/correlation_models.py').read_bytes()
p.seed(seed);base=p.CorrelationModel(text_in_dim=768,vision_in_dim=47,dropout=.1).cuda();assert p.state_sha(base)==p.state_sha(model)
records,meta=p.load('mosi','train');assert meta==old['train'];dev,dm=p.load('mosi','dev');assert dm==old['dev']
model.eval();base.eval();batch=p.batch(records,np.arange(4),58,'cuda')
with torch.no_grad():assert torch.allclose(p.encode(model,batch),p.encode(base,batch),atol=1e-6)
dropout_values=sorted(set(float(m.p) for m in model.modules() if isinstance(m,torch.nn.Dropout)));assert dropout_values==[p.P['regularization']['dropout']]
attention_values=sorted(set(float(m.dropout) for m in model.modules() if isinstance(m,torch.nn.MultiheadAttention)));assert attention_values==dropout_values
p.save_json(D/'validation.json',dict(passed=True,identical_initialization=True,identical_train_dev=True,eval_forward_unchanged=True,dropout_values=dropout_values,attention_dropout_values=attention_values,weight_decay=p.P['regularization']['weight_decay'],reference=str(ref)))
del base,model;torch.cuda.empty_cache();print('VALIDATION_PASSED',D.name,flush=True)
p.run('mosi','same_length')
oldhist=json.loads((ref/'runs/mosi_same_length/history.json').read_text());newhist=json.loads((D/'runs/mosi_same_length/history.json').read_text())
for a,b in zip(oldhist,newhist):
 for k in ['epoch','anchor_order_sha256','train_donors_sha256']:assert a[k]==b[k]
for k in ['random','same_length','lengths']:assert np.array_equal(np.load(ref/'runs/mosi_same_length/dev_donors.npz')[k],np.load(D/'runs/mosi_same_length/dev_donors.npz')[k])
model=p.CorrelationModel(text_in_dim=768,vision_in_dim=47,dropout=p.P['regularization']['dropout']).cuda();rows=[]
for split,r in [('train',records),('dev',dev)]:
 lengths=np.array([x['length'] for x in r]);maps={'same_length':[p.donors(lengths,'same_length',p.P['dev_negative_seed']+k) for k in range(5)]}
 for ck in ['best','last']:
  model.load_state_dict(torch.load(D/('runs/mosi_same_length/'+ck+'_correlation.pt'),map_location='cuda'),strict=True)
  emb=p.embeddings(model,r,58,'cuda');report=p.report(emb,maps);rows.append(dict(split=split,checkpoint=ck,report=report))
p.save_json(D/'train_dev_summary.json',dict(paired_actual_orders_and_maps_passed=True,rows=rows))
for k,h in json.loads((D/'frozen_source.json').read_text()).items():assert p.sha(D/k)==h
p.save_json(D/'COMPLETE.json',dict(complete=True,source_unchanged=True,paired_validation=True,test_evaluated=False))
print('COMPLETE',D.name,flush=True)
