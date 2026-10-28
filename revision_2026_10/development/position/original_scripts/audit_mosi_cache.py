from pathlib import Path
import sys,json,pickle,importlib.util
import numpy as np,torch
from transformers import BertModel,BertTokenizer
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'scripts'))
from pilot import sha,save_json,CACHE
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 path=Path('/data01/lyw/GLoMo/datasets/mosi.pkl');bert=Path('/data01/lyw/GLoMo/BERT_EN')
 spec=importlib.util.spec_from_file_location('word_pipeline','/data01/lyw/GLoMo_repair_20261019T001/Correlation_Pretraining/modality_correlation/fixed_word_features.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with path.open('rb') as f:source=pickle.load(f)
 tokenizer=BertTokenizer.from_pretrained(str(bert),local_files_only=True);model=BertModel.from_pretrained(str(bert),local_files_only=True).cuda().eval();results={}
 for split in ['train','dev']:
  rows=source[split];records=torch.load(CACHE/'mosi'/(split+'.fixed_contextual_word_v1.pt'),map_location='cpu')['records'];assert len(rows)==len(records)
  for src,r in zip(rows,records):
   (words,visual,audio),label,sid=src;assert m.sample_id(sid)==r['sample_id'];_,owners=m._tokenize_words(words,tokenizer,60);kept=list(dict.fromkeys(owners));assert kept==r['kept_word_indices']
   for key,data in [('audio',audio),('visual',visual)]:assert torch.equal(torch.as_tensor(np.asarray(data)[kept],dtype=torch.float32),r[key]),(r['sample_id'],key)
  lengths=np.array([r['length'] for r in records]);rng=np.random.RandomState(20261023);ids=sorted(set([0,len(rows)-1,int(lengths.argmin()),int(lengths.argmax())]+rng.choice(len(rows),12,replace=False).tolist()))
  tokenized=[m._tokenize_words(rows[i][0][0],tokenizer,60) for i in ids];t=max(len(x[0]) for x in tokenized);input_ids=torch.full((len(ids),t),tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(input_ids)
  for j,(tokens,owners) in enumerate(tokenized):input_ids[j,:len(tokens)]=torch.tensor(tokens,device='cuda');mask[j,:len(tokens)]=1
  hidden=model(input_ids=input_ids,attention_mask=mask).last_hidden_state.cpu();errors=[]
  for j,i in enumerate(ids):
   _,owners=tokenized[j];kept=list(dict.fromkeys(owners));pooled=torch.stack([hidden[j,[k+1 for k,v in enumerate(owners) if v==wi]].mean(0) for wi in kept]);error=float((pooled-records[i]['text']).abs().max());assert error<2e-5,error;errors.append(error)
  results[split]={'all_records':len(records),'sample_order_word_owners_audio_visual_exact':True,'text_recomputed_samples':len(ids),'text_sample_ids':[records[i]['sample_id'] for i in ids],'text_max_absolute_error':max(errors)}
 save_json(R/'mosi_cache_audit.json',{'passed':True,'scope':'MOSI train/dev feature lineage only; all audio/visual and word-owner mappings checked; frozen BERT text recomputed on deterministic subset. No test examples evaluated.','pkl_sha256':sha(path),'results':results});print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
