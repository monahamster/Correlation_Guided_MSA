import hashlib, types
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def align(examples,cache,data,tokenizer,maxlen):
    maps=[]
    for i,(ex,rec) in enumerate(zip(examples,cache['records'])):
        words,v,a=ex[0]
        owners=[j for j,w in enumerate(words) for _ in tokenizer.tokenize(w)][:maxlen-2]
        kept=list(dict.fromkeys(owners)); assert kept==list(rec['kept_word_indices'])
        assert len(kept)==rec['length']>0
        lut={w:j for j,w in enumerate(kept)}
        ix=np.asarray([lut[w] for w in owners],dtype=np.int64)
        for ti,name,raw in [(1,'visual',v),(2,'audio',a)]:
            vals=np.asarray(rec[name],dtype=np.float32)
            assert np.array_equal(vals,np.asarray(raw,dtype=np.float32)[kept])
            assert np.array_equal(data.tensors[ti][i,1:len(ix)+1].numpy(),vals[ix])
            assert not torch.count_nonzero(data.tensors[ti][i,0])
            assert not torch.count_nonzero(data.tensors[ti][i,len(ix)+1:])
        maps.append(ix)
    return maps

def corrupt(data,cache,maps,ids,ds,kind,level,seed,guided):
    if kind=='clean':return data,dict(input_digest='clean',words_affected_fraction=0.)
    ti,ci,name=(2,7,'audio') if kind=='audio' else (1,8,'visual')
    ts=list(data.tensors);ts[ti]=ts[ti].clone()
    if guided:ts[ci]=ts[ci].clone()
    digest=hashlib.sha256();affected=0;total=0
    for i,(rec,ix,sid) in enumerate(zip(cache['records'],maps,ids)):
        key=(ds+'|'+str(sid)+'|'+kind+'|'+str(seed)).encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))
        vals=np.asarray(rec[name],dtype=np.float32).copy()
        n=len(vals);total+=n
        if kind=='audio':
            vals += np.float32(level)*rng.standard_normal(vals.shape).astype(np.float32)
            affected+=n
        else:
            drop=rng.random(n)<level; vals[drop]=0.;affected+=int(drop.sum())
        assert np.isfinite(vals).all()
        digest.update(str(sid).encode());digest.update(vals.tobytes())
        ts[ti][i,1:len(ix)+1]=torch.from_numpy(vals[ix])
        assert torch.equal(ts[ti][i,0],data.tensors[ti][i,0])
        assert torch.equal(ts[ti][i,len(ix)+1:],data.tensors[ti][i,len(ix)+1:])
        if guided:
            ts[ci][i,:n]=torch.from_numpy(vals)
            assert torch.equal(ts[ti][i,1:len(ix)+1],ts[ci][i,torch.from_numpy(ix)])
            assert torch.equal(ts[ci][i,n:],data.tensors[ci][i,n:])
    for j,t in enumerate(ts):
        if j not in [ti,ci if guided else -1]:assert t is data.tensors[j]
    return torch.utils.data.TensorDataset(*ts),dict(input_digest=digest.hexdigest(),words_affected_fraction=affected/total)

class Recorder:
    def __init__(self,model):
        self.buffers={};self.max_gate_error=0.;self.originals=[]
        for name in ['t','a','v']:
            mod=getattr(model,'moe_'+name)
            original=mod.noisy_top_k_gating
            self.originals.append((mod,original))
            def wrapper(this,x,train,noise_epsilon=1e-2,reliability=None,_name=name,_original=original):
                assert not train
                gates,load=_original(x,train,noise_epsilon,reliability=reliability)
                base=torch.nan_to_num(x,nan=0,posinf=0,neginf=0) @ this.w_gate
                bias=torch.zeros_like(base)
                if this.use_reliability and reliability is not None:
                    rr=reliability.unsqueeze(1) if reliability.dim()==1 else reliability
                    bias=this.reliability_lambda*(rr @ this.w_rel)
                logits=base+bias
                top,idx=logits.topk(min(this.k+1,this.num_experts),dim=1)
                expected=torch.zeros_like(logits).scatter(1,idx[:,:this.k],this.softmax(top[:,:this.k]))
                err=(expected-gates).abs().max().item()
                self.max_gate_error=max(self.max_gate_error,err)
                assert err<1e-6,(err,_name)
                for key,val in [('gates',gates),('base_logits',base),('rel_bias',bias)]:
                    self.buffers.setdefault(_name+'_'+key,[]).append(val.detach().cpu().numpy())
                return gates,load
            mod.noisy_top_k_gating=types.MethodType(wrapper,mod)
    def reset(self):self.buffers={}
    def collect(self,n):
        out={k:np.concatenate(v) for k,v in self.buffers.items()}
        assert len(out)==9 and all(len(v)==n for v in out.values())
        return out

def gates_np(logits,k=2):
    # No tied top-k boundary for saved logits is expected; count ties separately.
    indices=np.argsort(-logits,axis=1,kind='stable')[:,:k]
    top=np.take_along_axis(logits,indices,axis=1);w=np.exp(top-top.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    g=np.zeros_like(logits);np.put_along_axis(g,indices,w,axis=1)
    return g

def routing_summary(arr,clean):
    out={}
    for mod in ['t','a','v']:
        g=arr[mod+'_gates'];cg=clean[mod+'_gates']
        base=arr[mod+'_base_logits'];bias=arr[mod+'_rel_bias']
        cf=gates_np(base+clean[mod+'_rel_bias'])
        logits=base+bias
        sorted_logits=np.sort(logits,axis=1)
        out[mod]=dict(mean_gate=g.mean(0).tolist(),entropy=float(-(g*np.log(np.maximum(g,1e-30))).sum(1).mean()),
            top1_switch_fraction=float((g.argmax(1)!=cg.argmax(1)).mean()),
            top2_set_switch_fraction=float(((g>0)!=(cg>0)).any(1).mean()),
            gate_l1_from_clean=float(np.abs(g-cg).sum(1).mean()),
            reliability_bias_l1_from_clean=float(np.abs(bias-clean[mod+'_rel_bias']).sum(1).mean()),
            fixed_feature_reliability_gate_l1=float(np.abs(g-cf).sum(1).mean()),
            fixed_feature_reliability_top2_switch=float(((g>0)!=(cf>0)).any(1).mean()),
            topk_boundary_tie_fraction=float((sorted_logits[:,0]==sorted_logits[:,1]).mean()))
    for key in arr:
        if key.startswith(('q_','s_','raw_cosine_')):
            out[key]=dict(mean=float(arr[key].mean()),mean_delta_clean=float((arr[key]-clean[key]).mean()))
    return out
