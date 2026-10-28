from pathlib import Path
import os,sys,json,subprocess,shlex
import torch
import torch.nn.functional as F
R=Path(sys.argv[1]);os.chdir(R/'Correlation_Guided_MSA');sys.path.insert(0,str(Path.cwd()));torch.set_num_threads(4);os.environ['PYTHONHASHSEED']='5576'
cmd=subprocess.check_output([sys.executable,str(R/'scripts/run_formal.py'),'mosei','dynamic','5576','--gpu','0'],text=True);sys.argv=shlex.split(cmd.splitlines()[1])[3:]
import main_cgmsa as m
from formal_protocol import tensor_digest
m.validate_formal_args();p=json.loads((R/'protocol/protocol.json').read_text());m.set_random_seed(5576);model=m.prep_for_training(2);model.eval();before=tensor_digest(model.corr_model.state_dict().items())
cache=torch.load(Path(p['cache_root'])/'mosei/dev.fixed_contextual_word_v1.pt',map_location='cpu');records=cache['records'][:16]
mask=torch.zeros(16,78,dtype=torch.bool,device=m.DEVICE);data={k:torch.zeros(16,78,d,device=m.DEVICE) for k,d in [('text',768),('audio',74),('visual',35)]}
for i,row in enumerate(records):
 n=row['length'];mask[i,:n]=True
 for k in data:data[k][i,:n]=torch.as_tensor(row[k],device=m.DEVICE)
actual=model._correlation_signals(data['text'],data['audio'],data['visual'],mask)
with torch.no_grad():
 encoded=model.corr_model(data['text'],data['audio'],data['visual'],mask,mask,mask);rows=[]
 for i in range(16):
  local=[]
  for a,b in [(0,1),(0,2),(1,2)]:
   aa=encoded[a][i,mask[i]];bb=encoded[b][i,mask[i]]
   if p['correlation_readout']=='word':v=F.cosine_similarity(aa,bb,dim=-1).clamp(-1,1).mean()
   else:v=F.cosine_similarity(aa.mean(0),bb.mean(0),dim=0)
   local.append(v)
  rows.append(torch.stack(local))
 expected=torch.stack(rows);observed=torch.stack([actual[k] for k in ['raw_cosine_ta','raw_cosine_tv','raw_cosine_av']],1);error=float((expected-observed).abs().max());assert torch.allclose(expected,observed,atol=2e-6)
 c=(expected+1)/2;q=torch.stack([(c[:,0]+c[:,1])/2,(c[:,0]+c[:,2])/2,(c[:,1]+c[:,2])/2],1);scales=1+.2*(q-.5)
 assert torch.allclose(torch.stack([actual[k] for k in ['q_t','q_a','q_v']],1),q,atol=2e-6)
 assert torch.allclose(torch.stack([actual[k] for k in ['s_t','s_a','s_v']],1),scales,atol=2e-6)
assert tensor_digest(model.corr_model.state_dict().items())==before
report=dict(passed=True,real_dev_samples=16,readout=p['correlation_readout'],max_cosine_error=error,q_and_scales_match_manual=True,correlation_weights_unchanged=True)
(R/'protocol/readout_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
