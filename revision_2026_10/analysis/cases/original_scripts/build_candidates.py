import json,pickle,re,hashlib,datetime
from pathlib import Path
import numpy as np,h5py,torch
from transformers import BertTokenizer
tokenizer=BertTokenizer.from_pretrained("/data01/lyw/GLoMo/BERT_EN",local_files_only=True)
R=Path('/data01/lyw/GLoMo_case_analysis_20261022')
F=Path('/data01/lyw/GLoMo_final_evaluation_20261022')
B=Path('/data01/lyw/GLoMo_robustness_20261022')
SEEDS=[5576,42,2026,2027,3407]; VARS=['baseline','constant','dynamic']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,obj): (R/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def sid(meta):return meta if isinstance(meta,str) else '_'.join(str(x) for x in meta)
protocol={'created_utc':datetime.datetime.utcnow().isoformat(),'stage':'posthoc_descriptive_case_analysis','seeds':SEEDS,'routing_seed':5576,'selection':'Four pools: consistent dynamic gain versus both controls; consistent loss versus both; all-model agreement; largest prediction spread. Four candidates per pool per dataset, unique video IDs within each dataset. Candidate review only, not final six cases. All test records retained.','eligibility':'Valid nonnegative timestamps and 11-character video ID; 3 <= clip seconds <= 25, 5 <= word count <= 60, cache retains every original word and downstream subwords are not truncated; gains/losses require >=4/5 paired seeds versus each control.','blind_annotation':'Separate worksheet omits labels, model outputs and selection groups. No claim of independent blind annotation by the selecting analyst.','media_evidence':'Do not annotate voice/facial cues until corresponding media is observed.','checkpoint_selection':'Existing frozen dev-selected checkpoints; no test selection or retraining.','timestamp_sources':{'mosi':'Original label CSD intervals by zero-based segment index, label agreement verified','mosei':'GLoMo PKL metadata, exact video/start/end; no external media synchronization verified yet'}}
dump('protocol.json',protocol)
allrows=[];checks={};sources={};short=[]
csdp=Path('/data01/lyw/GLoMo-main/datasets/cmumosi/CMU_MOSI_Labels.csd')
csd=h5py.File(csdp,'r');sources[str(csdp)]=sha(csdp)
for ds in ['mosi','mosei']:
 pk=Path('/data01/lyw/GLoMo/datasets')/(ds+'.pkl'); examples=pickle.load(open(pk,'rb'))['test'];sources[str(pk)]=sha(pk)
 cp=Path('/data01/lyw/GLoMo_repair_20261019T001/repair_cache')/ds/'test.fixed_contextual_word_v1.pt'
 cache=torch.load(cp,map_location='cpu')['records'];sources[str(cp)]=sha(cp)
 ids=np.array([sid(x[2]) for x in examples]);labels=np.array([float(np.asarray(x[1]).ravel()[0]) for x in examples]); assert len(set(ids))==len(ids)
 preds={};trace={}
 for v in VARS:
  pp=[]
  for seed in SEEDS:
   p=F/'evaluations/test'/f'{ds}_{v}_seed{seed}'/'predictions.npz';z=np.load(p);sources[str(p)]=sha(p)
   assert np.array_equal(z['sample_ids'],ids) and np.array_equal(z['labels'],labels)
   pp.append(z['predictions'])
  preds[v]=np.stack(pp)
  p=B/'evaluations'/f'{ds}_{v}_seed5576'/'clean_0.0_seed0.npz';z=np.load(p);sources[str(p)]=sha(p);trace[v]={k:z[k] for k in z.files}
  assert np.array_equal(z['sample_ids'],ids) and np.array_equal(z['predictions'],preds[v][0])
 errors={v:np.abs(preds[v]-labels) for v in VARS}; rows=[]; label_err=[]
 for i,(ex,rec) in enumerate(zip(examples,cache)):
  words=list(ex[0][0]); subwords=sum(len(tokenizer.tokenize(w)) for w in words);assert sid(ex[2])==str(rec['sample_id']); kept=[int(x) for x in rec['kept_word_indices']]
  if ds=='mosi':
   vid,seg=re.fullmatch(r'(.+)\[(\d+)\]',ids[i]).groups();g=csd['Opinion Segment Labels/data/'+vid];a,b=map(float,g['intervals'][int(seg)]); label_err.append(abs(float(g['features'][int(seg)].ravel()[0])-labels[i]))
  else: vid=str(ex[2][0]);a=float(ex[2][1]);b=float(ex[2][2])
  assert b>a
  timestamp_valid=a>=0; youtube_id_valid=bool(re.fullmatch(r"[A-Za-z0-9_-]{11}",vid))
  row={'dataset':ds,'index':i,'sample_id':str(ids[i]),'video_id':vid,'start_seconds':a,'end_seconds':b,'duration':b-a,'timestamp_valid':timestamp_valid,'youtube_id_valid':youtube_id_valid,'transcript':' '.join(words),'word_count':len(words),'subword_count':subwords,'all_subwords_retained':subwords<=({'mosi':60,'mosei':80}[ds]-2),'model_input_text':' '.join(words[j] for j in kept),'all_words_retained':kept==list(range(len(words))),'kept_word_indices':kept,'label':float(labels[i]),'public_video_url':('https://www.youtube.com/watch?v='+vid+'&t='+str(max(0,int(a)))+'s') if youtube_id_valid else None,'media_status':'not_acquired','models':{}}
  for v in VARS:
   row['models'][v]={'predictions_by_seed':preds[v][:,i].tolist(),'prediction_mean':float(preds[v][:,i].mean()),'prediction_sd':float(preds[v][:,i].std(ddof=1)),'mean_absolute_error':float(errors[v][:,i].mean()),'seed5576_trace':{k:trace[v][k][i].tolist() for k in trace[v] if k not in ['sample_ids','predictions','labels']}}
  gains={v:errors[v][:,i]-errors['dynamic'][:,i] for v in ['baseline','constant']}
  row['dynamic_gain']={v:{'mean':float(g.mean()),'wins':int((g>0).sum()),'losses':int((g<0).sum())} for v,g in gains.items()}
  row['prediction_mean_spread']=float(np.ptp([row['models'][v]['prediction_mean'] for v in VARS]));rows.append(row)
 assert len(rows)==len(cache)==len(examples)
 if label_err: assert max(label_err)<1e-6
 checks[ds]={'samples':len(rows),'all_prediction_ids_labels_match':True,'all_clean_trace_predictions_match':True,'all_cache_ids_match':True,'nonnegative_timestamp_count':sum(r['timestamp_valid'] for r in rows),'negative_start_samples':[r['sample_id'] for r in rows if not r['timestamp_valid']],'non_youtube_id_count':sum(not r['youtube_id_valid'] for r in rows),'mosi_csd_max_label_error':max(label_err) if label_err else None,'fully_retained_samples':sum(r['all_words_retained'] for r in rows)}
 eligible=[r for r in rows if r['timestamp_valid'] and r['youtube_id_valid'] and 3<=r['duration']<=25 and 5<=r['word_count']<=60 and r['all_words_retained'] and r['all_subwords_retained']]
 pools={
 'consistent_gain':sorted([r for r in eligible if all(r['dynamic_gain'][v]['wins']>=4 for v in ['baseline','constant'])],key=lambda r:-min(r['dynamic_gain'][v]['mean'] for v in ['baseline','constant'])),
 'consistent_loss':sorted([r for r in eligible if all(r['dynamic_gain'][v]['losses']>=4 for v in ['baseline','constant'])],key=lambda r:max(r['dynamic_gain'][v]['mean'] for v in ['baseline','constant'])),
 'agreement':sorted(eligible,key=lambda r:(r['prediction_mean_spread'],r['sample_id'])),
 'prediction_spread':sorted(eligible,key=lambda r:-r['prediction_mean_spread'])}
 used=set(); counts={}
 for group,pool in pools.items():
  counts[group]={'eligible':len(pool),'selected':0}
  for r in pool:
   if r['video_id'] in used:continue
   r['candidate_group']=group;r['candidate_id']=ds+'_'+group+'_'+str(counts[group]['selected']+1);short.append(r);used.add(r['video_id']);counts[group]['selected']+=1
   if counts[group]['selected']==4:break
 checks[ds]['candidate_pools']=counts;allrows.extend(rows)
 print(ds,json.dumps(checks[ds]),flush=True)
csd.close()
with open(R/'all_test_cases.jsonl','w') as f:
 for r in allrows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
dump('candidates.json',short);dump('alignment_validation.json',checks);dump('source_hashes.json',sources)
print('CANDIDATES',len(short),flush=True)
