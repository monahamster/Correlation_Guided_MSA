from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;Q=R/'replications'
rows=[];checks=[]
for seed in [5576,42,2026]:
 if seed==5576:dirs={'sentence':Path('/data01/lyw/GLoMo_position_ablation_20261020'),'word_sentence':R/'word_sentence_triplet'}
 else:dirs={a:Q/('seed{}_{}'.format(seed,a)) for a in ['sentence','word_sentence']}
 data={}
 for arm,d in dirs.items():
  for k,v in json.loads((d/'frozen_source.json').read_text()).items():assert hashlib.sha256((d/k).read_bytes()).hexdigest()==v
  s=json.loads((d/'runs/mosi_same_length/summary.json').read_text());v=s['dev']['same_length'];i=json.loads((d/'runs/mosi_same_length/inputs.json').read_text());h=json.loads((d/'runs/mosi_same_length/history.json').read_text());assert len(h)==20
  data[arm]=(i,h);rows.append(dict(seed=seed,role='development' if seed==5576 else 'confirmation',arm=arm,best_epoch=s['best_epoch'],dev_loss=v['loss'],auc={k:x['auc_mean'] for k,x in v['pairs'].items()}))
 for k in ['initial_state_sha256','train','dev']:assert data['sentence'][0][k]==data['word_sentence'][0][k]
 for a,b in zip(data['sentence'][1],data['word_sentence'][1]):
  for k in ['epoch','anchor_order_sha256','train_donors_sha256']:assert a[k]==b[k]
 a=np.load(dirs['sentence']/'runs/mosi_same_length/dev_donors.npz');b=np.load(dirs['word_sentence']/'runs/mosi_same_length/dev_donors.npz')
 for k in a.files:assert np.array_equal(a[k],b[k])
 checks.append(dict(seed=seed,initialization_data_order_donors_identical=True))
deltas=[]
for seed in [5576,42,2026]:
 a=next(x for x in rows if x['seed']==seed and x['arm']=='sentence');b=next(x for x in rows if x['seed']==seed and x['arm']=='word_sentence')
 deltas.append(dict(seed=seed,dev_loss_word_minus_sentence=b['dev_loss']-a['dev_loss'],auc_word_minus_sentence={k:b['auc'][k]-a['auc'][k] for k in a['auc']}))
report=dict(scope='Seed5576 development plus prespecified confirmation seeds42/2026; three training seeds distinct from five eval donor maps; no test; same sentence validation selection in both arms.',paired_checks=checks,rows=rows,deltas=deltas)
(Q/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
