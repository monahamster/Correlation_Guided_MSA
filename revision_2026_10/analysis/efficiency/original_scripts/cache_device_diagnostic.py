import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,pickle,json,sys
from pathlib import Path
from transformers import BertModel,BertTokenizer
torch.set_num_threads(4)
root=Path('/data01/lyw/GLoMo_repair_20261019T001');R=Path('/data01/lyw/GLoMo_efficiency_20261022')
sys.path.insert(0,str(root/'Correlation_Pretraining/modality_correlation'))
from fixed_word_features import _tokenize_words
with (root/'datasets/mosi.pkl').open('rb') as f:examples=pickle.load(f)['train']
old=torch.load(root/'repair_cache/mosi/train.fixed_contextual_word_v1.pt',map_location='cpu')['records']
new=torch.load(R/'cache_rebuilt/mosi/train.fixed_contextual_word_v1.pt',map_location='cpu')['records']
tokenizer=BertTokenizer.from_pretrained(str(root/'BERT_EN'));model=BertModel.from_pretrained(str(root/'BERT_EN')).eval()
rows=[]
for device in ['cpu','cuda']:
 model=model.to(device)
 for start in [448,464]:
  chunk=examples[start:start+16];tokens=[_tokenize_words(ex[0][0],tokenizer,60) for ex in chunk];longest=max(len(x[0]) for x in tokens)
  ids=torch.zeros((len(chunk),longest),dtype=torch.long,device=device);mask=torch.zeros_like(ids)
  for i,(row,_) in enumerate(tokens):ids[i,:len(row)]=torch.tensor(row,device=device);mask[i,:len(row)]=1
  with torch.no_grad():hidden=model(input_ids=ids,attention_mask=mask).last_hidden_state.cpu()
  for i,(_,owners) in enumerate(tokens):
   kept=list(dict.fromkeys(owners));text=torch.stack([hidden[i,[j+1 for j,owner in enumerate(owners) if owner==wi]].mean(0) for wi in kept])
   rows.append(dict(device=device,index=start+i,old_error=float((text-old[start+i]['text']).abs().max()),new_error=float((text-new[start+i]['text']).abs().max())))
(R/'cache_device_diagnostic.json').write_text(json.dumps(rows,indent=2))
print(json.dumps([r for r in rows if r['index'] in [458,478]],indent=2),flush=True)
