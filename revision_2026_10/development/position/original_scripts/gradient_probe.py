from pathlib import Path
import sys,json,importlib.util
import numpy as np,torch
import torch.nn.functional as F
R=Path(__file__).resolve().parent;OLD=Path('/data01/lyw/GLoMo_length_pretrain_20261020');sys.path.insert(0,str(R/'scripts'))
from pilot import load,seed,P,CorrelationModel,batch,encode,donors,loss_rows,save_json

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.enabled=False
 spec=importlib.util.spec_from_file_location('reference_position',OLD/'scripts/correlation_models.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 records,_=load('mosi','train');lengths=np.array([r['length'] for r in records]);d=donors(lengths,'same_length',20261024);order=np.random.RandomState(20261024).permutation(len(records))[:256];rows=[]
 for name,root,cls in [('with_position',OLD,m.CorrelationModel),('without_position',R,CorrelationModel)]:
  for checkpoint in ['initial','last']:
   seed(P['seed']);model=cls(text_in_dim=768,vision_in_dim=47).cuda()
   if checkpoint=='last':model.load_state_dict(torch.load(root/'runs/mosi_same_length/last_correlation.pt',map_location='cuda:0'))
   for mode in ['eval','train']:
    model.train(mode=='train');stats=[]
    for start in range(0,len(order),32):
     idx=order[start:start+32];seed(20261024+start);pos=encode(model,batch(records,idx,58,'cuda'));neg=encode(model,batch(records,d[idx],58,'cuda'));loss=loss_rows(pos,neg).mean();model.zero_grad(set_to_none=True);loss.backward()
     grads=[p.grad.detach() for p in model.parameters() if p.grad is not None];gn=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt());pn=float(torch.stack([p.detach().square().sum() for p in model.parameters()]).sum().sqrt())
     gaps=torch.stack([F.cosine_similarity(pos[:,a],pos[:,b])-F.cosine_similarity(pos[:,a],neg[:,b]) for a in range(3) for b in range(3) if a!=b],1)
     stats.append({'loss':float(loss),'gradient_l2':gn,'gradient_relative_to_parameter_l2':gn/pn,'active_triplet_fraction':float((gaps<P['margin']).float().mean()),'positive_minus_negative_mean':float(gaps.mean())})
    summary={k:float(np.mean([v[k] for v in stats])) for k in stats[0]};rows.append({'position':name,'checkpoint':checkpoint,'mode':mode,'means':summary,'batches':stats});print(name,checkpoint,mode,summary,flush=True)
   del model;torch.cuda.empty_cache()
 save_json(R/'mosi_gradient_probe.json',{'scope':'Eight fixed train batches; same anchors/donors; no optimizer steps. Eval disables dropout while retaining gradients. Norms describe local optimization, not generalization.','rows':rows})
if __name__=='__main__':main()
