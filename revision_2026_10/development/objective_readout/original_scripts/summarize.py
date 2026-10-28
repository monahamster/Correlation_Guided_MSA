from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parents[1];entries=json.loads((R/'run_matrix.json').read_text());by={(e['seed'],e['objective']):Path(e['root'])/'runs/mosei_same_length' for e in entries};checks={}
for seed in [5576,42,2026]:
 a=by[seed,'sentence'];b=by[seed,'word_sentence'];ia=json.loads((a/'inputs.json').read_text());ib=json.loads((b/'inputs.json').read_text());ha=json.loads((a/'history.json').read_text());hb=json.loads((b/'history.json').read_text());assert len(ha)==len(hb)==20
 c={k:ia[k]==ib[k] for k in ['train','dev','initial_state_sha256']}
 for k in ['anchor_order_sha256','train_donors_sha256']:c[k]=[h[k] for h in ha]==[h[k] for h in hb]
 x=np.load(a/'dev_donors.npz');y=np.load(b/'dev_donors.npz');c['dev_maps']=all(np.array_equal(x[k],y[k]) for k in x.files)
 assert all(c.values()),c;checks[str(seed)]=c
rows=[]
for row in json.loads((R/'evidence/mosi_existing_readout.json').read_text())['rows']:
 if row['checkpoint']=='best':rows.append(dict(row,dataset='mosi'))
for e in entries:rows.append(json.loads((R/'evidence'/('mosei_seed{}_{}.json'.format(e['seed'],e['objective']))).read_text()))
summary={}
for dataset in ['mosi','mosei']:
 summary[dataset]={}
 for family in ['sentence','word_sentence']:
  group=[x for x in rows if x['dataset']==dataset and x['family']==family];assert sorted(x['seed'] for x in group)==[42,2026,5576]
  out={}
  for metric in ['sentence_auc','word_average_auc']:
   out[metric]={k:{'mean':float(np.mean([x['pairs'][k][metric] for x in group])),'sample_sd':float(np.std([x['pairs'][k][metric] for x in group],ddof=1)),'per_seed':{str(x['seed']):x['pairs'][k][metric] for x in group}} for k in ['text_audio','text_visual','audio_visual']}
  summary[dataset][family]=out
report={'scope':'Three pretrained seeds per dataset; both objectives and both readouts at existing sentence-loss-selected checkpoints. MOSI no extra positions; MOSEI with positions. AUC uses utterances; five donor maps averaged within seed. Final configuration selection awaits downstream dev comparison.','paired_mosei_checks':checks,'summary':summary,'rows':[{k:v for k,v in x.items() if k!='data'} for x in rows]}
(R/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');(R/'paired_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(summary,indent=2))
