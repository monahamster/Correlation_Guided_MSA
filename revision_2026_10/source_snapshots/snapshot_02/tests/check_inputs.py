import sys,json,pickle,gc
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Correlation_Guided_MSA'))
from formal_protocol import strict_cache,verify_inputs
policy=json.loads((ROOT/'protocol/protocol.json').read_text())
report={}
def sid(x):
 if isinstance(x,np.ndarray):x=x.tolist()
 return '_'.join(map(str,x)) if isinstance(x,(tuple,list)) else str(x)
for ds in ['mosi','mosei']:
 verify_inputs(ROOT,ds)
 with (ROOT/'datasets'/(ds+'.pkl')).open('rb') as f:data=pickle.load(f)
 result={}
 for split in ['train','dev','test']:
  cache,path,_=strict_cache(ROOT/'datasets'/(ds+'.pkl'),ds,split,ROOT/'BERT_EN',policy['cache_root'],policy['datasets'][ds]['max_seq_length'])
  records=cache['records'];ids=[sid(x[2]) for x in data[split]]
  assert ids==[r['sample_id'] for r in records] and len(set(ids))==len(ids)
  for rec in records:
   for key,dim in [('text',768),('audio',74),('visual',policy['datasets'][ds]['VISUAL_DIM'])]:
    assert rec[key].shape==(rec['length'],dim)
    assert rec[key].isfinite().all()
   assert 0<rec['length']<=policy['datasets'][ds]['max_seq_length']-2
  result[split]={'samples':len(ids),'id_order_exact':True,'duplicate_ids':0,'finite_features_and_dimensions':True}
  del cache,records;gc.collect()
 report[ds]=result
 print('PASSED input fingerprints and all cache records:',ds,flush=True)
 del data;gc.collect()
(ROOT/'protocol/input_validation.json').write_text(json.dumps(report,indent=2))
