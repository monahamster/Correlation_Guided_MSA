from pathlib import Path
import sys,json,importlib.util
import numpy as np,torch
import torch.nn.functional as F
R=Path(__file__).resolve().parent;OLD=Path('/data01/lyw/GLoMo_length_pretrain_20261020');sys.path.insert(0,str(R/'scripts'))
from pilot import load,seed,P,CorrelationModel,embeddings,donors,report,save_json,sha,state_sha
@torch.no_grad()
def run(ds):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.enabled=False
 spec=importlib.util.spec_from_file_location('with_position',OLD/'scripts/correlation_models.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 data={};maps={}
 for split in ['train','dev']:
  data[split],_=load(ds,split);lengths=np.array([r['length'] for r in data[split]])
  maps[split]={a:[donors(lengths,a,P['dev_negative_seed']+k) for k in range(5)] for a in P['arms']}
 rows=[];out=R/'generalization'/ds;out.mkdir(parents=True,exist_ok=False)
 for label,root,cls in [('with_position',OLD,mod.CorrelationModel),('without_position',R,CorrelationModel)]:
  folder=root/'runs'/(ds+'_same_length');assert json.loads((folder/'status.json').read_text())['state']=='complete'
  for checkpoint in ['initial','best','last']:
   seed(P['seed']);model=cls(text_in_dim=768,vision_in_dim=47 if ds=='mosi' else 35).cuda().eval();model.requires_grad_(False)
   if checkpoint=='initial':assert state_sha(model)==json.loads((folder/'inputs.json').read_text())['initial_state_sha256']
   else:model.load_state_dict(torch.load(folder/(checkpoint+'_correlation.pt'),map_location='cuda:0'),strict=True)
   row={'position':label,'checkpoint':checkpoint,'epoch':0 if checkpoint=='initial' else (json.loads((folder/'summary.json').read_text())['best_epoch'] if checkpoint=='best' else 20),'splits':{}}
   for split in ['train','dev']:
    e=embeddings(model,data[split],P['padding'][ds],'cuda:0');norm=F.normalize(e,dim=-1).numpy();center=norm-norm.mean(0,keepdims=True);concentration={}
    for k,m in enumerate(['text','audio','visual']):
     v=np.linalg.svd(center[:,k],compute_uv=False)**2;v=v/max(v.sum(),1e-30)
     concentration[m]={'different_sample_cosine':float((norm[:,k]*norm[maps[split]['random'][0],k]).sum(-1).mean()),'centered_effective_rank':float(np.exp(-(v[v>0]*np.log(v[v>0])).sum()))}
    row['splits'][split]={'n':len(e),'pairing':report(e,maps[split]),'concentration':concentration}
    np.save(out/(label+'_'+checkpoint+'_'+split+'.npy'),e.numpy())
   rows.append(row);print(ds,label,checkpoint,'train/dev losses',*[round(row['splits'][s]['pairing']['same_length']['loss'],5) for s in ['train','dev']],flush=True)
   del model;torch.cuda.empty_cache()
 save_json(out/'summary.json',{'scope':'Post-hoc train/dev evaluation to distinguish optimization failure from generalization failure. Same evaluation donor generator; no test and no parameter updates.','dataset':ds,'rows':rows})
if __name__=='__main__':run(sys.argv[1])
