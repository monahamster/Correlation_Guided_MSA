from pathlib import Path
import argparse,json,os,sys,pickle,datetime,time,hashlib
R=Path(__file__).resolve().parents[1]
F=Path('/data01/lyw/GLoMo_final_evaluation_20261022')
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--preflight',action='store_true')
o=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(o.gpu);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,torch
from torch.utils.data import DataLoader
from robust_utils import sha,align,corrupt,Recorder,routing_summary
from metrics import sentiment_metrics
torch.set_num_threads(4)
protocol=json.loads((R/'protocol.json').read_text())
e=next(e for e in json.loads((R/'evaluation_manifest.json').read_text())['entries'] if e['id']==o.id)
root=Path(e['root']);run=Path(e['run']);ds=e['dataset'];split='dev' if o.preflight else 'test'
out=R/('preflight' if o.preflight else 'evaluations')/o.id
out.mkdir(parents=True,exist_ok=False)
assert sha(run/'run_config.json')==e['config_sha256']
assert sha(e['checkpoint'])==e['checkpoint_sha256']
for name,digest in e['source_files'].items():assert sha(root/name)==digest,name
cfg=json.loads((run/'run_config.json').read_text());a=cfg['arguments']
data_path=(root/'datasets'/(ds+'.pkl')).resolve()
for path in [data_path,(root/'BERT_EN/pytorch_model.bin').resolve()]:
    assert sha(path)==e['input_sha256'][str(path)]
assert not a['test_audio_noise_std'] and not a['test_visual_mask_ratio']
assert not a['load'] and not a['paired_init_load'] and not a['paired_init_save']
os.chdir(root/'Correlation_Guided_MSA');sys.path.insert(0,str(Path.cwd()))
sys.argv=[sys.argv[0],'--dataset',ds,'--VISUAL_DIM',str(a['VISUAL_DIM']),'--d_l',str(a['d_l']),'--max_seq_length',str(a['max_seq_length'])]
import main_cgmsa as m
m.args=argparse.Namespace(**a);m.set_random_seed(e['seed'])
with data_path.open('rb') as f:examples=pickle.load(f)[split]
assert len(examples)==e['samples'][split]
# Load cache for alignment even for baseline; it is never passed into baseline model.
cache_root=a['corr_cache_root'] or '/data01/lyw/GLoMo_repair_20261019T001/repair_cache'
cache=m.strict_cache(str(data_path),ds,split,'../BERT_EN/',cache_root,a['max_seq_length'])[0]
cachepath=Path(cache_root)/ds/(split+'.fixed_contextual_word_v1.pt')
assert sha(cachepath)==e['input_sha256'][str(cachepath)]
if o.preflight:
    examples=examples[:64];cache={**cache,'records':cache['records'][:64]}
guided=a['use_fusion_correlation'] or a['use_moe_reliability']
data=m.get_appropriate_dataset(examples,cache if guided else None)
ids=np.asarray([m.stringify_sample_id(x[2]) for x in examples])
assert list(ids)==[r['sample_id'] for r in cache['records']]
maps=align(examples,cache,data,m.get_tokenizer(a['model']),a['max_seq_length'])
model=m.prep_for_training(1)
model.load_state_dict(torch.load(e['checkpoint'],map_location=m.DEVICE),strict=True);model.eval()
if a['correlation_control']=='constant':
    assert np.allclose(model.constant_scales.detach().cpu().numpy(),cfg['constant_scales'],atol=1e-7)
def forward(d,recorder=None):
    pred=[];labels=[];analysis={}
    if recorder:recorder.reset()
    with torch.no_grad():
        for batch in DataLoader(d,batch_size=a[split+'_batch_size'],shuffle=False,num_workers=0):
            ids_,v,au,mask,seg,y,kw=m.unpack_batch(tuple(t.to(m.DEVICE) for t in batch))
            result=model.test(ids_,v.squeeze(1),au.squeeze(1),attention_mask=mask,token_type_ids=seg,labels=None,return_analysis=recorder is not None,**kw)
            pred.append(result[0].reshape(-1).cpu().numpy());labels.append(y.reshape(-1).cpu().numpy())
            if recorder:
                for key,val in result[3].items():
                    if key.startswith(('q_','s_','raw_cosine_')) and val is not None:
                        analysis.setdefault(key,[]).append(val.detach().cpu().numpy().reshape(-1))
    arr=dict(predictions=np.concatenate(pred),labels=np.concatenate(labels),sample_ids=ids)
    arr.update({k:np.concatenate(v) for k,v in analysis.items()})
    if recorder:arr.update(recorder.collect(len(ids)))
    return arr
unhooked=forward(data) if o.preflight else None
recorder=Recorder(model)
cases=[('clean',0.,0)]
seeds=protocol['corruption_seeds'][:1] if o.preflight else protocol['corruption_seeds']
for kind,key in [('audio','audio_noise_std'),('visual','visual_word_mask_probability')]:
    levels=protocol[key][-1:] if o.preflight else protocol[key][1:]
    cases.extend((kind,float(level),s) for level in levels for s in seeds)
rows=[];clean=None;t0=time.time()
for index,(kind,level,seed) in enumerate(cases):
    d,audit=corrupt(data,cache,maps,ids,ds,kind,level,seed,guided)
    arr=forward(d,recorder)
    assert np.isfinite(arr['predictions']).all()
    if kind=='clean':
        clean=arr
        saved=np.load(run/'dev_predictions.npz' if o.preflight else F/'evaluations/test'/o.id/'predictions.npz')
        n=len(ids);assert np.array_equal(saved['labels'][:n],arr['labels'])
        if 'sample_ids' in saved:assert np.array_equal(saved['sample_ids'][:n],ids)
        diff=float(np.max(np.abs(saved['predictions'][:n]-arr['predictions'])))
        assert np.allclose(saved['predictions'][:n],arr['predictions'],atol=2e-5,rtol=1e-5),diff
        if unhooked:assert np.array_equal(unhooked['predictions'],arr['predictions'])
    else:assert np.array_equal(arr['labels'],clean['labels'])
    name=kind+'_'+str(level)+'_seed'+str(seed)
    np.savez_compressed(out/(name+'.npz'),**arr)
    row=dict(condition=name,kind=kind,level=level,corruption_seed=seed,metrics=sentiment_metrics(arr['predictions'],arr['labels']),routing=routing_summary(arr,clean),input_audit=audit)
    row['mae_degradation']=row['metrics']['mae']-sentiment_metrics(clean['predictions'],clean['labels'])['mae']
    rows.append(row)
    status=dict(id=o.id,dataset=ds,seed=e['seed'],split=split,completed=len(rows),total=len(cases),rows=rows,status='complete' if len(rows)==len(cases) else 'running',checkpoint_sha256=e['checkpoint_sha256'],protocol_sha256=sha(R/'protocol.json'),clean_replay_max_abs_diff=diff,gate_reconstruction_max_abs_diff=recorder.max_gate_error,unhooked_equality_passed=o.preflight,elapsed_seconds=time.time()-t0,updated_utc=datetime.datetime.utcnow().isoformat()+'Z')
    tmp=out/'progress.tmp';tmp.write_text(json.dumps(status,indent=2,allow_nan=False));tmp.replace(out/'progress.json')
    print('CONDITION_COMPLETE',o.id,index+1,len(cases),kind,level,seed,flush=True)
print('ROBUSTNESS_COMPLETE',o.id,flush=True)
