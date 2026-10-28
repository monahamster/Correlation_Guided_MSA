import collections
import torch
from torch import nn

class DenseFlops:
    """Forward-shape dense arithmetic count; no parameter-value assumptions."""
    def __init__(self,model):
        self.rows=[];self.handles=[];skip=[]
        for name,mod in model.named_modules():
            if any(name.startswith(p+'.') for p in skip):continue
            cls=mod.__class__.__name__
            atomic=isinstance(mod,(nn.TransformerEncoderLayer,nn.MultiheadAttention)) or cls in ['MultiheadAttention','BertSelfAttention']
            if atomic:skip.append(name)
            if atomic or isinstance(mod,(nn.Linear,nn.Conv1d)) or cls in ['MoE','Attention']:
                self.handles.append(mod.register_forward_hook(self.hook(name),with_kwargs=True))
    def hook(self,name):
        def fn(mod,args,kwargs,out):
            cls=mod.__class__.__name__;x=args[0] if args else kwargs.get('query',kwargs.get('hidden_states',kwargs.get('src')));shape=list(x.shape);f=0;kind=cls
            if isinstance(mod,nn.TransformerEncoderLayer):
                b,l,d=x.shape if mod.self_attn.batch_first else (x.shape[1],x.shape[0],x.shape[2])
                ff=mod.linear1.out_features
                f=8*b*l*d*d+4*b*l*l*d+4*b*l*d*ff
            elif cls=='BertSelfAttention':
                b,l,d=x.shape
                assert mod.all_head_size==d
                f=6*b*l*d*d+4*b*l*l*d
            elif isinstance(mod,nn.MultiheadAttention) or cls=='MultiheadAttention':
                q,k,v=args[:3] if len(args)>=3 else (kwargs['query'],kwargs['key'],kwargs['value']);assert k.shape==v.shape
                bf=getattr(mod,'batch_first',False);b=q.shape[0 if bf else 1];lq=q.shape[1 if bf else 0];lk=k.shape[1 if bf else 0];d=q.shape[2]
                assert k.shape[2]==d
                f=2*b*(lq+2*lk)*d*d+2*b*lq*d*d+4*b*lq*lk*d
            elif isinstance(mod,nn.Linear):
                f=2*out.numel()*mod.in_features
            elif isinstance(mod,nn.Conv1d):
                f=2*out.numel()*(mod.in_channels//mod.groups)*mod.kernel_size[0]
            elif cls=='MoE':
                b=x.shape[0];d,e=mod.w_gate.shape;f=2*b*d*e
                if mod.use_reliability:f+=2*b*mod.w_rel.shape[0]*e
            elif cls=='Attention':
                f=2*x.shape[0]*x.shape[1]*3
            else:raise AssertionError(cls)
            self.rows.append(dict(module=name,kind=kind,input_shape=shape,flops=int(f)))
        return fn
    def clear(self):self.rows=[]
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]
    def result(self):
        by=collections.defaultdict(int)
        for r in self.rows:by[r['module'].split('.')[0]]+=r['flops']
        return dict(total_flops=sum(r['flops'] for r in self.rows),by_module=dict(by),operators=list(self.rows))

def self_check():
    # Independent closed-form checks of atomic encoder + plain projection.
    layer=nn.TransformerEncoderLayer(d_model=8,nhead=2,dim_feedforward=16,batch_first=True,dropout=0).eval()
    net=nn.Sequential(layer,nn.Linear(8,4)).eval();counter=DenseFlops(net)
    with torch.no_grad():net(torch.ones(2,3,8))
    got=counter.result()['total_flops'];expected=8*2*3*8**2+4*2*3**2*8+4*2*3*8*16+2*2*3*8*4
    counter.close();assert got==expected,(got,expected)
    return dict(synthetic_encoder_projection_flops=got,passed=True)
