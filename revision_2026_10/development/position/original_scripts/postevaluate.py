from pathlib import Path
import sys,json
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from pilot import load,sha,save_json,CorrelationModel,P
sys.path.insert(0,'/data01/lyw/GLoMo_diagnostics_20261020/scripts')
from correlation_signals import encode,signals,describe,MODS
@torch.no_grad()
def run(ds,arm):
 torch.set_num_threads(4);torch.manual_seed(20261020)
 folder=ROOT/'runs'/(ds+'_'+arm);out=folder/'postdiagnostics';out.mkdir(exist_ok=False)
 assert json.loads((folder/'status.json').read_text())['state']=='complete'
 records,meta=load(ds,'dev');clean=np.load(folder/'best_dev_embeddings.npy');null=np.load(folder/'zero_dev_embeddings.npy')
 model=CorrelationModel(text_in_dim=768,vision_in_dim=47 if ds=='mosi' else 35).cuda().eval();model.requires_grad_(False)
 model.load_state_dict(torch.load(folder/'best_correlation.pt',map_location='cuda:0'),strict=True)
 direct=np.stack([encode(model,records,m,'cuda:0') for m in MODS],1);error=float(np.abs(direct-clean).max());assert error<1e-5,error
 cos,q,s=signals(clean);_,nq,ns=signals(null)
 config=json.loads(Path('/data01/lyw/GLoMo_diagnostics_20261020/diagnostic_protocol.json').read_text())
 response={};arrays={'clean_cos':cos,'clean_q':q,'clean_s':s,'zero_q':nq,'zero_s':ns}
 for condition in config['conditions']:
  k=MODS.index(condition['modality']);changed=clean.copy();changed[:,k]=encode(model,records,condition['modality'],'cuda:0',condition=condition)
  cc,qq,ss=signals(changed);delta=ss[:,k]-s[:,k]
  response[condition['name']]={'affected_modality':condition['modality'],'affected_scale_delta':describe(delta),'fraction_scale_decreases':float((delta<0).mean()),'all_scale_delta_mean':(ss-s).mean(0).tolist()}
  arrays[condition['name']]=ss
 d=np.load(folder/'dev_donors.npz')['random'][0];norm=F.normalize(torch.as_tensor(clean),dim=-1).numpy();concentration={}
 for k,m in enumerate(MODS):
  concentration[m]={'different_sample_cosine':describe((norm[:,k]*norm[d,k]).sum(-1)),'clean_q':describe(q[:,k]),'zero_q':describe(nq[:,k]),'scale_std':float(s[:,k].std(ddof=1))}
 summary={'dataset':ds,'arm':arm,'eligible_dev_samples':len(records),'checkpoint_sha256':sha(folder/'best_correlation.pt'),'reused_protocol':'GLoMo_diagnostics_20261020; same perturbation construction and seed on eligible dev subset','conditions':config['conditions'],'separate_vs_full_forward_max_error':error,'corruption_signal_response':response,'concentration':concentration,'scope':'Branch-only controlled feature perturbations; no sentiment robustness result.'}
 save_json(out/'summary.json',summary);np.savez_compressed(out/'signals.npz',**arrays)
 print(ds,arm,'postdiagnostics complete',flush=True)
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
