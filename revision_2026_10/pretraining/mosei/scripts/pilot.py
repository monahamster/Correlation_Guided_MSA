from pathlib import Path
import sys,json,hashlib,random,time,os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from correlation_models import CorrelationModel
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/data01/lyw/GLoMo_repair_20261019T001/repair_cache')
MODS=('text','audio','visual')
PAIRS=((0,1),(0,2),(1,2))
P=json.loads((ROOT/'protocol.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save_json(path,obj):
 q=path.with_suffix(path.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(path)
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.cpu().numpy().tobytes())
 return h.hexdigest()
def load(ds,split):
 path=CACHE/ds/(split+'.fixed_contextual_word_v1.pt');cache=torch.load(path,map_location='cpu')
 expected={x['path']:x['sha256'] for x in json.loads(Path('/data01/lyw/GLoMo_formal_20261020/protocol/input_manifest.json').read_text())['files']}
 actual=sha(path);assert actual==expected[str(path)],'Input hash changed'
 allrec=cache['records'];lengths=np.array([r['length'] for r in allrec]);u,c=np.unique(lengths,return_counts=True)
 ids=np.flatnonzero(np.isin(lengths,u[c>=2]));eligible_set=set(ids.tolist());records=[allrec[i] for i in ids]
 assert len({r['sample_id'] for r in allrec})==len(allrec)
 for r in allrec:
  assert 0<r['length']<=P['padding'][ds]
  for k in MODS:assert len(r[k])==r['length'] and torch.isfinite(torch.as_tensor(r[k])).all()
 return records,{'path':str(path),'sha256':actual,'total':len(allrec),'eligible':len(records),'excluded_ids':[r['sample_id'] for i,r in enumerate(allrec) if i not in eligible_set],'eligible_ids':[r['sample_id'] for r in records]}
def donors(lengths,arm,nseed):
 rng=np.random.RandomState(nseed);n=len(lengths);d=np.full(n,-1,dtype=int)
 groups=[np.arange(n)] if arm=='random' else [np.flatnonzero(lengths==l) for l in np.unique(lengths)]
 for g in groups:
  assert len(g)>1
  q=rng.permutation(g);d[q]=np.roll(q,int(rng.randint(1,len(q))))
 assert np.all(d>=0) and np.all(d!=np.arange(n)) and len(np.unique(d))==n
 if arm=='same_length':assert np.array_equal(lengths,lengths[d])
 return d
def batch(records,indices,maxlen,device,zero=False):
 out={k:torch.zeros(len(indices),maxlen,torch.as_tensor(records[0][k]).shape[1],device=device) for k in MODS}
 mask=torch.zeros(len(indices),maxlen,dtype=torch.bool,device=device)
 for j,i in enumerate(indices):
  r=records[i];n=r['length'];mask[j,:n]=True
  if not zero:
   for k in MODS:out[k][j,:n]=torch.as_tensor(r[k],device=device)
 return out,mask
def encode(model,b):
 x,m=b;y=model(x['text'],x['audio'],x['visual'],m,m,m)
 return torch.stack([(v*m[:,:,None]).sum(1)/m.sum(1)[:,None] for v in y],1)
def loss_rows(pos,neg):
 terms=[]
 for a in range(3):
  for b in range(3):
   if a!=b:terms.append(F.relu(F.cosine_similarity(pos[:,a],neg[:,b])-F.cosine_similarity(pos[:,a],pos[:,b])+P['margin']))
 return torch.stack(terms,1).sum(1)/3
@torch.no_grad()
def embeddings(model,records,maxlen,device,zero=False):
 model.eval();out=[]
 for start in range(0,len(records),P['batch_size']):out.append(encode(model,batch(records,np.arange(start,min(start+P['batch_size'],len(records))),maxlen,device,zero)).cpu())
 return torch.cat(out)
def report(emb,maps):
 n=len(emb);norm=F.normalize(emb,dim=-1);out={}
 for arm,dd in maps.items():
  pairs={}
  for a,b in PAIRS:
   pos=(norm[:,a]*norm[:,b]).sum(-1).numpy();neg=np.stack([(norm[:,a]*norm[d,b]).sum(-1).numpy() for d in dd])
   auc=[float(roc_auc_score(np.r_[np.ones(n),np.zeros(n)],np.r_[pos,v])) for v in neg]
   pairs[MODS[a]+'_'+MODS[b]]={'positive_mean':float(pos.mean()),'negative_mean':float(neg.mean()),'auc_each':auc,'auc_mean':float(np.mean(auc))}
  out[arm]={'loss':float(np.mean([loss_rows(emb,emb[d]).mean().item() for d in dd])),'pairs':pairs}
 return out
def run(ds,arm):
 assert ds in P['datasets'] and arm in P['arms']
 folder=ROOT/'runs'/(ds+'_'+arm);folder.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.enabled=False;torch.use_deterministic_algorithms(True,warn_only=True)
 device='cuda:0';seed(P['seed']);maxlen=P['padding'][ds]
 train,trainmeta=load(ds,'train');dev,devmeta=load(ds,'dev')
 assert set(trainmeta['eligible_ids']).isdisjoint(devmeta['eligible_ids'])
 lengths=np.array([r['length'] for r in train]);dl=np.array([r['length'] for r in dev])
 maps={a:[donors(dl,a,P['dev_negative_seed']+k) for k in range(5)] for a in P['arms']}
 np.savez_compressed(folder/'dev_donors.npz',**{a:np.stack(d) for a,d in maps.items()},lengths=dl)
 model=CorrelationModel(text_in_dim=768,vision_in_dim=47 if ds=='mosi' else 35).to(device)
 initial=state_sha(model)
 save_json(folder/'inputs.json',{'train':trainmeta,'dev':devmeta,'initial_state_sha256':initial,'protocol_sha256':sha(ROOT/'protocol.json'),'source_sha256':{p.name:sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))},'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
 optim=torch.optim.Adam(model.parameters(),lr=P['learning_rate']);history=[];best=float('inf');started=time.time()
 for epoch in range(1,P['epochs']+1):
  d=donors(lengths,arm,P['seed']+P['train_negative_seed_offset']+epoch)
  order=np.random.RandomState(P['seed']+P['anchor_order_seed_offset']+epoch).permutation(len(train))
  model.train();total=0.;count=0
  for step,start in enumerate(range(0,len(train),P['batch_size'])):
   seed(P['seed']+P['dropout_seed_offset']+epoch*10000+step)
   idx=order[start:start+P['batch_size']];pos=encode(model,batch(train,idx,maxlen,device));neg=encode(model,batch(train,d[idx],maxlen,device))
   loss=loss_rows(pos,neg).mean()
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   optim.zero_grad(set_to_none=True);loss.backward()
   if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():raise ValueError('Nonfinite gradient')
   optim.step();total+=loss.item()*len(idx);count+=len(idx)
  emb=embeddings(model,dev,maxlen,device);stats=report(emb,maps);criterion=stats['same_length']['loss']
  row={'epoch':epoch,'train_loss':total/count,'dev':stats,'elapsed_seconds':time.time()-started,'anchor_order_sha256':hashlib.sha256(order.tobytes()).hexdigest(),'train_donors_sha256':hashlib.sha256(d.tobytes()).hexdigest()};history.append(row)
  if criterion<best:
   best=criterion;best_epoch=epoch;torch.save(model.state_dict(),folder/'best_correlation.pt');np.save(folder/'best_dev_embeddings.npy',emb.numpy())
  save_json(folder/'history.json',history);save_json(folder/'status.json',{'state':'running','epoch':epoch,'best_epoch':best_epoch,'best_dev_loss':best,'elapsed_seconds':time.time()-started})
  print(json.dumps({'dataset':ds,'arm':arm,'epoch':epoch,'train_loss':row['train_loss'],'dev_loss':criterion,'auc':[stats['same_length']['pairs'][MODS[a]+'_'+MODS[b]]['auc_mean'] for a,b in PAIRS]}),flush=True)
 torch.save(model.state_dict(),folder/'last_correlation.pt')
 model.load_state_dict(torch.load(folder/'best_correlation.pt',map_location=device),strict=True)
 clean=embeddings(model,dev,maxlen,device);null=embeddings(model,dev,maxlen,device,zero=True)
 assert np.allclose(clean.numpy(),np.load(folder/'best_dev_embeddings.npy'),atol=2e-6)
 zero_report={m:float(F.cosine_similarity(clean[:,i],null[:,i]).mean()) for i,m in enumerate(MODS)}
 np.save(folder/'zero_dev_embeddings.npy',null.numpy())
 save_json(folder/'summary.json',{'dataset':ds,'arm':arm,'best_epoch':best_epoch,'initial_state_sha256':initial,'best_checkpoint_sha256':sha(folder/'best_correlation.pt'),'dev':report(clean,maps),'all_zero_dev':report(null,maps),'clean_zero_embedding_cosine_mean':zero_report,'elapsed_seconds':time.time()-started,'train_eligible':len(train),'dev_eligible':len(dev),'scope':'Exploratory train/dev pretraining pilot only; not a sentiment result.'})
 save_json(folder/'status.json',{'state':'complete','epochs':len(history),'best_epoch':best_epoch,'elapsed_seconds':time.time()-started})
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
