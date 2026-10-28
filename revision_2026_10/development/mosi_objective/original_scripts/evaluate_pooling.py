from pathlib import Path
import sys,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;D=ROOT/sys.argv[1];sys.path.insert(0,str(D/'scripts'))
import pilot as p
assert json.loads((D/'runs/mosi_same_length/status.json').read_text())['state']=='complete'
for name,h in json.loads((D/'frozen_source.json').read_text()).items():assert p.sha(D/name)==h
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ref=Path(json.loads((D/'reference.json').read_text())['root']);old=json.loads((ref/'runs/mosi_same_length/history.json').read_text());new=json.loads((D/'runs/mosi_same_length/history.json').read_text())
for a,b in zip(old,new):
 for key in ['epoch','anchor_order_sha256','train_donors_sha256']:assert a[key]==b[key]
model=p.CorrelationModel(text_in_dim=768,vision_in_dim=47).cuda();rows=[]
for split in ['train','dev']:
 records,_=p.load('mosi',split);lengths=np.array([r['length'] for r in records]);maps={'same_length':[p.donors(lengths,'same_length',p.P['dev_negative_seed']+k) for k in range(5)]}
 for ck in ['best','last']:
  model.load_state_dict(torch.load(D/('runs/mosi_same_length/'+ck+'_correlation.pt'),map_location='cuda'))
  emb=p.embeddings(model,records,58,'cuda');report=p.report(emb,maps)
  rows.append(dict(split=split,checkpoint=ck,report=report));print(D.name,split,ck,'loss',report['same_length']['loss'],'auc',[v['auc_mean'] for v in report['same_length']['pairs'].values()],flush=True)
p.save_json(D/'train_dev_summary.json',dict(scope='Same eval donor maps as previous mean-pooling train/dev diagnostic; no test.',paired_training_maps_passed=True,rows=rows))
