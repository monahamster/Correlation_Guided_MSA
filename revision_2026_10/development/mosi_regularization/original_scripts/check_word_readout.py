from pathlib import Path
import sys,json
import torch,numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;S=Path('/data01/lyw/GLoMo_mosi_pretrain_opt_20261020');sys.path.insert(0,str(S/'word_sentence_triplet/scripts'))
import pilot as p
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
records,meta=p.load('mosi','dev');lengths=np.array([r['length'] for r in records]);mask=torch.arange(58)[None,:]<torch.from_numpy(lengths[:,None]);maps=[p.donors(lengths,'same_length',p.P['dev_negative_seed']+k) for k in range(5)];n=len(records)
rows=[];model=p.CorrelationModel(text_in_dim=768,vision_in_dim=47).cuda()
@torch.no_grad()
def encode():
 model.eval();out=[]
 for start in range(0,n,32):out.append(p.token_encode(model,p.batch(records,np.arange(start,min(start+32,n)),58,'cuda')).cpu())
 return torch.cat(out)
def auc(pos,neg):return float(roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(pos))],np.r_[pos,neg]))
for seed in [5576,42,2026]:
 d=S/'word_sentence_triplet' if seed==5576 else S/'replications'/('seed{}_word_sentence'.format(seed))
 for checkpoint in ['best','last']:
  model.load_state_dict(torch.load(d/('runs/mosi_same_length/'+checkpoint+'_correlation.pt'),map_location='cuda'),strict=True)
  z=encode();u=F.normalize(z,dim=-1);sentence=F.normalize(p.pool_tokens(z,mask),dim=-1);pairs={}
  for a,b in [(0,1),(0,2),(1,2)]:
   key=p.MODS[a]+'_'+p.MODS[b];pos_sentence=(sentence[:,a]*sentence[:,b]).sum(-1).numpy()
   pos_word=(((u[:,:,a]*u[:,:,b]).sum(-1)*mask).sum(1)/mask.sum(1)).numpy()
   sent_aucs=[];word_aucs=[]
   for donor in maps:
    neg_sentence=(sentence[:,a]*sentence[donor,b]).sum(-1).numpy();neg_word=(((u[:,:,a]*u[donor,:,b]).sum(-1)*mask).sum(1)/mask.sum(1)).numpy()
    sent_aucs.append(auc(pos_sentence,neg_sentence));word_aucs.append(auc(pos_word,neg_word))
   eligible=np.flatnonzero(lengths>1);shifted=u[:,:,b].clone()
   for i in eligible:shifted[i,:lengths[i]]=torch.roll(u[i,:lengths[i],b],1,0)
   pos=pos_word[eligible];neg=(((u[:,:,a]*shifted).sum(-1)*mask).sum(1)/mask.sum(1)).numpy()[eligible]
   pairs[key]=dict(sentence_auc=float(np.mean(sent_aucs)),word_average_auc=float(np.mean(word_aucs)),word_auc_each=word_aucs,aligned_word_cosine_mean=float(pos_word.mean()),within_sample_one_word_shift_auc=auc(pos,neg),alignment_drop_mean=float((pos-neg).mean()),shift_eligible=len(eligible))
  if checkpoint=='best':
   ref=json.loads((d/'runs/mosi_same_length/summary.json').read_text())['dev']['same_length']['pairs']
   for k in pairs:assert abs(pairs[k]['sentence_auc']-ref[k]['auc_mean'])<1e-6
  rows.append(dict(seed=seed,checkpoint=checkpoint,pairs=pairs));print(seed,checkpoint,[(k,round(v['sentence_auc'],4),round(v['word_average_auc'],4),round(v['alignment_drop_mean'],5)) for k,v in pairs.items()],flush=True)
p.save_json(R/'word_readout_diagnostic.json',dict(scope='Read-only MOSI dev diagnostic on existing word+sentence checkpoints. Candidate readout changes aggregation only: mean aligned-word cosine versus cosine of mean embeddings. Each AUC observation is an utterance, not an independent token. Five original same-length donor maps. One-word cyclic shift is a separate temporal alignment control. No retraining, labels or test.',data=meta,rows=rows))
