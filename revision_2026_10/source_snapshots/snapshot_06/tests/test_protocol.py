import sys,json,tempfile,random,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Correlation_Guided_MSA'))
sys.argv=[sys.argv[0]]
import main_cgmsa as main
from formal_protocol import isolated_rng,seeded_loader,strict_cache

torch.set_num_threads(2)
checks={}

def state():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
def equal(a,b):
    return a[0]==b[0] and all(np.array_equal(x,y) for x,y in zip(a[1],b[1])) and torch.equal(a[2],b[2])
a=state()
try:
    with isolated_rng():
        random.random();np.random.rand();torch.rand(4)
        raise ValueError('test exception')
except ValueError: pass
assert equal(a,state())
checks['rng_restored_even_on_exception']=True

data=TensorDataset(torch.arange(31))
x=seeded_loader(data,7,True,100)
y=seeded_loader(data,7,True,100)
a=state()
list(seeded_loader(data,4,False,999))
assert equal(a,state())
for epoch in range(2):
    assert torch.equal(torch.cat([b[0] for b in x]),torch.cat([b[0] for b in y]))
checks['independent_loaders_same_two_epoch_order']=True

class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.correlation_control='constant'
        self.register_buffer('constant_scales',torch.full((3,),float('nan')))
    def _correlation_signals(self,t,a,v,mask):
        return {'s_t':t[:,0,0],'s_a':a[:,0,0],'s_v':v[:,0,0]}
n=5
base=torch.zeros(n,2)
values=torch.arange(n,dtype=torch.float32).view(n,1,1)
ds=TensorDataset(base.long(),base,base,base.long(),base.long(),torch.zeros(n),values,values+1,values+2,torch.ones(n,1,dtype=torch.bool))
model=ConstantModel().train();a=state()
with patch.object(main.args,'correlation_control','constant'),patch.object(main.args,'dev_batch_size',2):
    result=main.set_training_scale_means(model,seeded_loader(ds,2,True,10))
assert result==[2,3,4] and model.training and model.correlation_control=='constant' and equal(a,state())
clone=ConstantModel();clone.load_state_dict(model.state_dict());assert torch.equal(clone.constant_scales,model.constant_scales)
checks['constant_full_train_mean_rng_and_checkpoint']=True

class ErrorModel(torch.nn.Module):
    def test(self,ids,*args,**kwargs): return ids[:,0].float(),None,None
err=torch.tensor([[1],[1],[4]])
errdata=TensorDataset(err,torch.zeros(3,1,1),torch.zeros(3,1,1),torch.ones_like(err),torch.zeros_like(err),torch.zeros(3))
mse,mae=main.eval_epoch(ErrorModel(),seeded_loader(errdata,2,False,1))
assert mae==2 and mse==6
checks['validation_sample_weighting']=True

with tempfile.TemporaryDirectory() as td:
    root=Path(td);m=torch.nn.Linear(1,1,bias=False);calls=[];opts=[]
    def epoch(model,loader,opt,epoch):
        calls.append('train');opts.append(id(opt))
        with torch.no_grad(): model.weight.fill_(epoch)
        return 1.0
    vals=iter([(2,1),(1,.5),(1,.5)])
    def evaluate(model,loader): calls.append('dev');return next(vals)
    def test(model,loader):
        assert calls==['train','dev']*3
        assert model.weight.item()==1
        calls.append('test');return [.1,.2,.3,.4,.5,.6,.7,.8]
    with patch.object(main,'train_epoch',epoch),patch.object(main,'eval_epoch',evaluate),patch.object(main,'test_score_model',test),patch.object(main,'resolve_experiment_dir',lambda:root),patch.object(main,'resolve_best_model_path',lambda:root/'best.pt'),patch.object(main,'resolve_metrics_path',lambda:root/'metrics.json'),patch.object(main.args,'n_epochs',3):
        result=main.train(m,None,None,None)
    assert result['best_epoch']==1 and len(set(opts))==1 and calls[-1]=='test'
    checks['single_optimizer_earliest_tie_final_test_selected_reload']=True

print(json.dumps(checks,indent=2))
(ROOT/'protocol/unit_validation.json').write_text(json.dumps(checks,indent=2))
