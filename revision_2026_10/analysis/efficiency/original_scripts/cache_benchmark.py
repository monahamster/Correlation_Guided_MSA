from pathlib import Path
import argparse,os,sys,json,time,datetime,gc,importlib.util,resource
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mosi','mosei'],required=True);p.add_argument('--gpu',type=int,default=0);o=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(o.gpu)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,numpy as np
from transformers import BertModel,BertConfig
from robust_utils import sha
torch.set_num_threads(4)
R=Path(__file__).resolve().parents[1];manifest=json.loads((R/'evaluation_manifest.json').read_text())
e=next(e for e in manifest['entries'] if e['id']==o.dataset+'_dynamic_seed5576')
root=Path(e['root']);cfg=json.loads((Path(e['run'])/'run_config.json').read_text())['arguments']
source=root/'Correlation_Pretraining/modality_correlation/fixed_word_features.py'
assert sha(source)==e['source_files'][str(source.relative_to(root))]
spec=importlib.util.spec_from_file_location('cache_source',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
bert=(root/'BERT_EN').resolve();pkl=(root/'datasets'/(o.dataset+'.pkl')).resolve()
assert sha(pkl)==e['input_sha256'][str(pkl)]
assert sha(bert/'pytorch_model.bin')==e['input_sha256'][str(bert/'pytorch_model.bin')]
model=BertModel(BertConfig.from_pretrained(str(bert)));generator_params=sum(p.numel() for p in model.parameters());del model;gc.collect()
output=R/'cache_rebuilt';result_path=R/('cache_'+o.dataset+'.json');assert not result_path.exists()
rows=[]
for split in ['train','dev','test']:
    original=Path(cfg['corr_cache_root'])/o.dataset/(split+'.fixed_contextual_word_v1.pt')
    assert sha(original)==e['input_sha256'][str(original)]
    target=mod.cache_path(output,o.dataset,split);assert not target.exists()
    torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    cache,path,created=mod.build_or_load_cache(str(pkl),o.dataset,split,str(bert),str(output),cfg['max_seq_length'],batch_size=16,device=torch.device('cuda'))
    torch.cuda.synchronize();elapsed=time.perf_counter()-t0
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    assert created
    t0=time.perf_counter();old=torch.load(original,map_location='cpu');load_s=time.perf_counter()-t0
    assert len(old['records'])==len(cache['records'])==e['samples'][split]
    diff=0.;sq=0.;ref_sq=0.;elements=0;words=0
    for actual,ref in zip(cache['records'],old['records']):
        for key in ['sample_id','length','kept_word_indices']:assert actual[key]==ref[key],(split,key)
        for key in ['audio','visual']:assert torch.equal(actual[key],ref[key]),(split,key)
        delta=(actual['text']-ref['text']).abs();diff=max(diff,float(delta.max()));sq+=float((delta.double()**2).sum());elements+=delta.numel();words+=actual['length']
        assert torch.isfinite(actual['text']).all()
        ref_sq+=float(ref['text'].double().square().sum())
    assert (sq/ref_sq)**.5<1e-4 and diff<0.01,(split,diff,(sq/ref_sq)**.5)
    row=dict(split=split,samples=len(cache['records']),retained_words=words,wall_seconds=elapsed,ms_per_sample=elapsed*1000/len(cache['records']),cache_file_bytes=path.stat().st_size,original_cache_bytes=original.stat().st_size,warm_filesystem_load_seconds=load_s,peak_gpu_allocated_bytes=peak,peak_gpu_reserved_bytes=reserved,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_text_replay_error=diff,text_replay_rmse=(sq/elements)**.5,text_relative_l2_error=(sq/ref_sq)**.5,text_bitwise_match=diff==0.,ids_word_maps_audio_visual_exact=True,rebuilt_cache_sha256=sha(path),source_cache_sha256=sha(original))
    rows.append(row)
    result=dict(dataset=o.dataset,status='complete' if len(rows)==3 else 'running',generator_parameters=generator_params,generator_fp32_weight_bytes=generator_params*4,batch_size=16,seq_length=cfg['max_seq_length'],source_sha256=sha(source),protocol_sha256=sha(R/'protocol.json'),splits=rows,updated_utc=datetime.datetime.utcnow().isoformat()+'Z')
    result_path.write_text(json.dumps(result,indent=2,allow_nan=False));print('CACHE_COMPLETE',o.dataset,split,elapsed,flush=True)
    del old,cache;gc.collect();torch.cuda.empty_cache()
