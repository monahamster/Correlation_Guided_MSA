"""Aligned-word readout copied from the validated standalone candidate."""
import torch
import torch.nn.functional as F

def score_encoded(encoded,mask,alpha=.2):
 if len(encoded)!=3:raise ValueError('Expected three modalities')
 if mask.dtype!=torch.bool or mask.ndim!=2:raise ValueError('Expected boolean [batch,word] valid mask')
 if any(x.shape[:2]!=mask.shape for x in encoded):raise ValueError('Encoded modalities must share aligned words')
 if not torch.all(mask.any(1)):raise ValueError('Each sample must have at least one valid word')
 cos=[]
 for a,b in [(0,1),(0,2),(1,2)]:
  local=F.cosine_similarity(encoded[a],encoded[b],dim=-1).clamp(-1,1)
  cos.append(local.masked_fill(~mask,0).sum(1)/mask.sum(1))
 cos=torch.stack(cos,1);c=(cos+1)/2
 q=torch.stack([(c[:,0]+c[:,1])/2,(c[:,0]+c[:,2])/2,(c[:,1]+c[:,2])/2],1)
 return {'cosines':cos,'q':q,'scales':1+alpha*(q-.5)}

