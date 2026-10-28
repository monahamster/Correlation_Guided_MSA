"""Reproducible correlation pretraining for the repair version only."""
import argparse, hashlib, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from correlation_models import CorrelationModel
from fixed_word_features import build_or_load_cache, CachedTripletDataset, collate_triplets

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def triplet(anchor, positive, negative, margin):
    return F.relu(F.cosine_similarity(anchor, negative, dim=-1) -
                  F.cosine_similarity(anchor, positive, dim=-1) + margin).mean()

def encode(model, batch, device):
    b = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
    ft, fa, fv = model(b["text"], b["audio"], b["visual"], b["mask"], b["mask"], b["mask"])
    w = b["mask"].float().unsqueeze(-1)
    return tuple((x*w).sum(1) / w.sum(1).clamp_min(1) for x in (ft,fa,fv))

def loss(model, pos, neg, device, margin):
    pt, pa, pv = encode(model, pos, device); nt, na, nv = encode(model, neg, device)
    value = (triplet(pa, pt, nt, margin) + triplet(pa, pv, nv, margin) +
             triplet(pt, pa, na, margin) + triplet(pt, pv, nv, margin) +
             triplet(pv, pa, na, margin) + triplet(pv, pt, nt, margin)) / 3.0
    stats = {"matched_ta": F.cosine_similarity(pt, pa, dim=-1).mean().item(),
             "matched_tv": F.cosine_similarity(pt, pv, dim=-1).mean().item(),
             "matched_av": F.cosine_similarity(pa, pv, dim=-1).mean().item(),
             "mismatched_ta": F.cosine_similarity(pt, na, dim=-1).mean().item()}
    return value, stats

def run_epoch(model, loader, device, margin, optimizer=None):
    train = optimizer is not None; model.train(train); totals=[]; stats=[]
    for pos, neg in loader:
        value, row = loss(model, pos, neg, device, margin)
        if train:
            optimizer.zero_grad(); value.backward(); optimizer.step()
        if not torch.isfinite(value): raise RuntimeError("non-finite triplet loss")
        totals.append(value.item()); stats.append(row)
    return float(np.mean(totals)), {k: float(np.mean([x[k] for x in stats])) for k in stats[0]}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset', choices=['mosi','mosei'], required=True)
    p.add_argument('--pkl-path', required=True); p.add_argument('--bert-path', required=True); p.add_argument('--cache-root', required=True)
    p.add_argument('--out-dir', required=True); p.add_argument('--max-seq-length', type=int, required=True)
    p.add_argument('--seed', type=int, default=5576); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--margin',type=float,default=.2); p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); set_seed(a.seed); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device(a.device)
    train_cache, train_path, _ = build_or_load_cache(a.pkl_path,a.dataset,'train',a.bert_path,a.cache_root,a.max_seq_length,device=device)
    dev_cache, dev_path, _ = build_or_load_cache(a.pkl_path,a.dataset,'dev',a.bert_path,a.cache_root,a.max_seq_length,device=device)
    visual = train_cache['records'][0]['visual'].shape[-1]; audio=train_cache['records'][0]['audio'].shape[-1]
    model=CorrelationModel(768,audio,visual,128,3,4,256,.1,64).to(device); opt=Adam(model.parameters(),lr=a.lr)
    fixed_dev=DataLoader(CachedTripletDataset(dev_cache,1),batch_size=a.batch_size,shuffle=False,collate_fn=collate_triplets)
    best=float('inf'); log=[]; best_path=out/'best_correlation.pt'
    for epoch in range(1,a.epochs+1):
        train_loader=DataLoader(CachedTripletDataset(train_cache,epoch),batch_size=a.batch_size,shuffle=True,collate_fn=collate_triplets)
        tl,ts=run_epoch(model,train_loader,device,a.margin,opt)
        with torch.no_grad(): vl,vs=run_epoch(model,fixed_dev,device,a.margin)
        row={'epoch':epoch,'train_loss':tl,'valid_loss':vl,'train_similarity':ts,'valid_similarity':vs}; log.append(row); print(json.dumps(row),flush=True)
        if vl < best:
            best=vl; torch.save(model.state_dict(),best_path)
    digest=hashlib.sha256(best_path.read_bytes()).hexdigest()
    (out/'metrics.json').write_text(json.dumps({'config':vars(a),'best_valid_loss':best,'best_checkpoint':str(best_path),'checkpoint_sha256':digest,'train_cache':str(train_path),'dev_cache':str(dev_path),'epochs':log},indent=2))
if __name__=='__main__': main()
