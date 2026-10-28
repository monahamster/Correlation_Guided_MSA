from pathlib import Path
import sys,json,re
import numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'scripts'))
from pilot import load,save_json
@torch.no_grad()
def run(ds):
 out=R/'generalization'/ds/'video_controls';out.mkdir(exist_ok=False)
 rows=[];video_sets={};coverage={}
 for split in ['train','dev']:
  records,_=load(ds,split);n=len(records);lengths=np.array([r['length'] for r in records]);videos=[]
  for rec in records:
   m=re.fullmatch(r'(.+)\[(\d+)\]',rec['sample_id']);assert m,rec['sample_id'];videos.append(m[1])
  videos=np.array(videos);video_sets[split]=set(videos);rng=np.random.RandomState(20261022)
  within={i:np.flatnonzero((lengths==lengths[i])&(videos==videos[i])&(np.arange(n)!=i)) for i in range(n)}
  across={i:np.flatnonzero((lengths==lengths[i])&(videos!=videos[i])) for i in range(n)}
  ids=np.array([i for i in range(n) if len(within[i]) and len(across[i])]);assert len(ids)>1
  maps={'same_video':np.array([[rng.choice(within[i]) for i in ids] for _ in range(5)]),'different_video':np.array([[rng.choice(across[i]) for i in ids] for _ in range(5)])}
  for mode,dd in maps.items():
   assert np.all(dd!=ids[None,:]);assert np.all(lengths[dd]==lengths[ids])
   assert np.all(videos[dd]==videos[ids]) if mode=='same_video' else np.all(videos[dd]!=videos[ids])
  coverage[split]={'total_eligible_length_records':n,'comparison_anchors':len(ids),'videos':len(set(videos)),'anchor_videos':len(set(videos[ids]))}
  np.savez_compressed(out/(split+'_donors.npz'),anchors=ids,**maps)
  for position in ['with_position','without_position']:
   for checkpoint in ['initial','best','last']:
    e=F.normalize(torch.from_numpy(np.load(R/'generalization'/ds/(position+'_'+checkpoint+'_'+split+'.npy'))),dim=-1).numpy();pairs={}
    for (a,b),key in zip([(0,1),(0,2),(1,2)],['text_audio','text_visual','audio_visual']):
     pos=(e[ids,a]*e[ids,b]).sum(-1);p={}
     for mode,dd in maps.items():
      neg=np.stack([(e[ids,a]*e[d,b]).sum(-1) for d in dd]);p[mode]={'auc_mean':float(np.mean([roc_auc_score(np.r_[np.ones(len(ids)),np.zeros(len(ids))],np.r_[pos,v]) for v in neg])),'positive_mean':float(pos.mean()),'negative_mean':float(neg.mean()),'matched_minus_negative_mean':float((pos[None]-neg).mean())}
     pairs[key]=p
    row={'split':split,'position':position,'checkpoint':checkpoint,'anchors':len(ids),'pairs':pairs};rows.append(row)
    if position=='without_position' and checkpoint=='last':print(ds,split,'anchors',len(ids),'same_video',[round(v['same_video']['auc_mean'],4) for v in pairs.values()],'different_video',[round(v['different_video']['auc_mean'],4) for v in pairs.values()],flush=True)
 save_json(out/'summary.json',{'dataset':ds,'scope':'Post-hoc source-video control. Identical positive anchors and exact lengths across negative conditions; random donor selection with replacement, five maps. Not a training change. Video ID is not assumed to equal speaker ID.','coverage':coverage,'train_dev_video_overlap':len(video_sets['train']&video_sets['dev']),'rows':rows})
if __name__=='__main__':run(sys.argv[1])
