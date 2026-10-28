from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1];entry=json.loads((R/'run_matrix.json').read_text())[int(sys.argv[1])];D=Path(entry['root']);sys.path.insert(0,str(D/'scripts'));import pilot as p
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.enabled=False;torch.use_deterministic_algorithms(True,warn_only=True)
folder=D/'runs/mosei_same_length';summary=json.loads((folder/'summary.json').read_text());checkpoint=folder/'best_correlation.pt';assert p.sha(checkpoint)==summary['best_checkpoint_sha256']
records,meta=p.load('mosei','dev');n=len(records);lengths=np.array([x['length'] for x in records]);mask=torch.arange(78)[None,:]<torch.from_numpy(lengths[:,None]);saved=np.load(folder/'dev_donors.npz');maps=[p.donors(lengths,'same_length',p.P['dev_negative_seed']+k) for k in range(5)];assert np.array_equal(np.stack(maps),saved['same_length'])
m=p.CorrelationModel(text_in_dim=768,vision_in_dim=35).cuda();m.load_state_dict(torch.load(checkpoint,map_location='cuda'),strict=True);m.eval();tokens=[];pooled=[]
with torch.no_grad():
 for start in range(0,n,32):
  data,valid=p.batch(records,np.arange(start,min(start+32,n)),78,'cuda');out=m(data['text'],data['audio'],data['visual'],valid,valid,valid);z=torch.stack(out,2)
  v=p.pool_tokens(z,valid) if hasattr(p,'pool_tokens') else torch.stack([(x*valid[:,:,None]).sum(1)/valid.sum(1)[:,None] for x in out],1)
  tokens.append(z.cpu());pooled.append(v.cpu())
z=torch.cat(tokens);sent=torch.cat(pooled);assert np.allclose(sent.numpy(),np.load(folder/'best_dev_embeddings.npy'),atol=2e-6)
u=F.normalize(z,dim=-1);s=F.normalize(sent,dim=-1)
def auc(a,b):return float(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b]))
pairs={}
for a,b in p.PAIRS:
 key=p.MODS[a]+'_'+p.MODS[b];pos_s=(s[:,a]*s[:,b]).sum(-1).numpy();pos_w=(((u[:,:,a]*u[:,:,b]).sum(-1)*mask).sum(1)/mask.sum(1)).numpy();sa=[];wa=[]
 for donor in maps:
  ns=(s[:,a]*s[donor,b]).sum(-1).numpy();nw=(((u[:,:,a]*u[donor,:,b]).sum(-1)*mask).sum(1)/mask.sum(1)).numpy();sa.append(auc(pos_s,ns));wa.append(auc(pos_w,nw))
 assert abs(np.mean(sa)-summary['dev']['same_length']['pairs'][key]['auc_mean'])<1e-6
 eligible=np.flatnonzero(lengths>1);shift=u[:,:,b].clone()
 for i in eligible:shift[i,:lengths[i]]=torch.roll(u[i,:lengths[i],b],1,0)
 shifted=(((u[:,:,a]*shift).sum(-1)*mask).sum(1)/mask.sum(1)).numpy()[eligible]
 pairs[key]=dict(sentence_auc=float(np.mean(sa)),word_average_auc=float(np.mean(wa)),sentence_auc_each=sa,word_auc_each=wa,aligned_word_cosine_mean=float(pos_w.mean()),within_sample_one_word_shift_auc=auc(pos_w[eligible],shifted),alignment_drop_mean=float((pos_w[eligible]-shifted).mean()),shift_eligible=len(eligible))
report=dict(dataset='mosei',seed=entry['seed'],family=entry['objective'],checkpoint='best',checkpoint_path=str(checkpoint),checkpoint_sha256=p.sha(checkpoint),selected_epoch=summary['best_epoch'],sentence_selection_loss=summary['dev']['same_length']['loss'],pairs=pairs,data=meta,original_sentence_auc_reproduced=True)
p.save_json(R/'evidence'/('mosei_seed{}_{}.json'.format(entry['seed'],entry['objective'])),report);print(json.dumps({k:v for k,v in report.items() if k!='data'},indent=2))
