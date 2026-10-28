from pathlib import Path
import argparse,json,os,sys,pickle,datetime,time,hashlib
R=Path(__file__).resolve().parents[1]
F=Path('/data01/lyw/GLoMo_final_evaluation_20261022')
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--pilot',action='store_true')
o=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(o.gpu);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,torch
from torch.utils.data import DataLoader
from robust_utils import sha,align,corrupt,Recorder,routing_summary
from dense_flops import DenseFlops,self_check
import gc,platform,subprocess
torch.set_num_threads(4)
protocol=json.loads((R/'protocol.json').read_text())
e=next(e for e in json.loads((R/'evaluation_manifest.json').read_text())['entries'] if e['id']==o.id)
root=Path(e['root']);run=Path(e['run']);ds=e['dataset'];split='dev'
out=R/('pilot' if o.pilot else 'evaluations')/o.id
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
guided=a['use_fusion_correlation'] or a['use_moe_reliability']
data=m.get_appropriate_dataset(examples,cache if guided else None)
ids=np.asarray([m.stringify_sample_id(x[2]) for x in examples])
assert list(ids)==[r['sample_id'] for r in cache['records']]
maps=align(examples,cache,data,m.get_tokenizer(a['model']),a['max_seq_length'])
model=m.prep_for_training(1)
model.load_state_dict(torch.load(e['checkpoint'],map_location=m.DEVICE),strict=True);model.eval()
if a['correlation_control']=='constant':
    assert np.allclose(model.constant_scales.detach().cpu().numpy(),cfg['constant_scales'],atol=1e-7)

def gpu_info():
    return subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader']).decode()
params=dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),fp32_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),by_top_module={})
for name,pv in model.named_parameters():
    key=name.split('.')[0];params['by_top_module'][key]=params['by_top_module'].get(key,0)+pv.numel()
params['correlation']=params['by_top_module'].get('corr_model',0)
rows=[];saved=np.load(run/'dev_predictions.npz');hook_errors=[];replay_errors=[]
validation=self_check();hardware_start=gpu_info()
# Replay the original evaluation batch shape before timing alternative batches.
with torch.no_grad():
    check_n=min(a['dev_batch_size'],len(data))
    check_batch=tuple(t[:check_n].to(m.DEVICE) for t in data.tensors)
    ci,cv,ca,cm,cs,cy,ck=m.unpack_batch(check_batch)
    check_pred=model.test(ci,cv.squeeze(1),ca.squeeze(1),attention_mask=cm,token_type_ids=cs,labels=None,**ck)[0].reshape(-1).cpu().numpy()
    validation['original_batch_replay_max_abs_diff']=float(np.abs(check_pred-saved['predictions'][:check_n]).max())
    assert np.allclose(check_pred,saved['predictions'][:check_n],atol=2e-5,rtol=1e-5)
    del check_batch,ci,cv,ca,cm,cs,cy,ck;gc.collect();torch.cuda.empty_cache()

with torch.no_grad():
    for bs in protocol['batch_sizes']:
        starts=np.linspace(0,len(data)-bs,protocol['input_blocks'],dtype=int).tolist()
        if o.pilot:starts=starts[:1]
        timing=[];corr_timing=[];memory=[];flops=[];profiles=[]
        for block,start in enumerate(starts):
            cpu=tuple(t[start:start+bs] for t in data.tensors)
            batch=tuple(t.to(m.DEVICE) for t in cpu)
            ids_,v,au,mask,seg,y,kw=m.unpack_batch(batch)
            def forward():
                return model.test(ids_,v.squeeze(1),au.squeeze(1),attention_mask=mask,token_type_ids=seg,labels=None,**kw)
            def corr_forward():return model._correlation_signals(**kw)
            pred=forward()[0].reshape(-1).cpu().numpy()
            expected=saved['predictions'][start:start+bs]
            err=float(np.abs(pred-expected).max());replay_errors.append(err)
            assert np.allclose(pred,expected,atol=1e-4,rtol=1e-5),(err,o.id,bs,start)
            # Hooks only for operation counting; remove before performance timing.
            counter=DenseFlops(model)
            instrumented=forward()[0].reshape(-1).cpu().numpy()
            counts=counter.result();counter.close()
            hook_errors.append(float(np.abs(instrumented-pred).max()))
            assert np.allclose(instrumented,pred,atol=1e-5,rtol=1e-5)
            counts.update(block=block,start=start,per_sample_flops=counts['total_flops']/bs)
            flops.append(counts)
            warm=2 if o.pilot else protocol['warmup_per_block']
            reps=3 if o.pilot else protocol['timed_repeats_per_block']
            for _ in range(warm):res=forward();del res
            torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
            resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            for _ in range(3):res=forward();del res
            torch.cuda.synchronize()
            memory.append(dict(block=block,resident_allocated_bytes=resident,peak_allocated_bytes=torch.cuda.max_memory_allocated(),incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident,peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            def measure(fn):
                wall=[];device=[]
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                for _ in range(reps):
                    torch.cuda.synchronize();t0=time.perf_counter();begin.record()
                    res=fn();end.record();torch.cuda.synchronize()
                    wall.append((time.perf_counter()-t0)*1000);device.append(begin.elapsed_time(end));del res
                return dict(block=block,start=start,wall_ms=wall,cuda_event_ms=device)
            timing.append(measure(forward))
            if guided:
                for _ in range(warm):res=corr_forward();del res
                corr_timing.append(measure(corr_forward))
            # Native profiler cross-check only after timing, on the first input block.
            if block==0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_flops=True) as prof:
                    res=forward();torch.cuda.synchronize();del res
                events=[dict(operator=x.key,count=x.count,flops=x.flops,input_shapes=x.input_shapes) for x in prof.key_averages(group_by_input_shape=True) if x.flops or any(z in x.key for z in ['attention','transformer_encoder','conv1d','addmm_activation'])]
                profiles=events
            del batch,ids_,v,au,mask,seg,y,kw,cpu;gc.collect();torch.cuda.empty_cache()
        def stats(ts):
            w=np.asarray([x for t in ts for x in t['wall_ms']]);g=np.asarray([x for t in ts for x in t['cuda_event_ms']])
            return dict(median_wall_ms=float(np.median(w)),p95_wall_ms=float(np.percentile(w,95)),median_cuda_ms=float(np.median(g)),samples_per_second=bs/(np.median(w)/1000),n_calls=len(w),per_block_wall_medians=[float(np.median(t['wall_ms'])) for t in ts])
        row=dict(batch_size=bs,sequence_length=a['max_seq_length'],correlation_padded_words=int(data.tensors[9].shape[1]) if guided else None,input_starts=starts,latency=stats(timing),correlation_latency=stats(corr_timing) if corr_timing else None,peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in memory),peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in memory),memory=memory,dense_flops_per_sample=float(np.mean([x['per_sample_flops'] for x in flops])),flop_details=flops,native_profiler=profiles,timing=timing,correlation_timing=corr_timing)
        rows.append(row)
        result=dict(id=o.id,dataset=ds,seed=e['seed'],status='complete' if len(rows)==len(protocol['batch_sizes']) else 'running',parameters=params,measurements=rows,validation=dict(synthetic=validation,max_saved_dev_replay_error=max(replay_errors),max_instrumentation_error=max(hook_errors)),environment=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),python=platform.python_version(),cpu_threads=torch.get_num_threads(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,hardware_start=hardware_start,hardware_latest=gpu_info()),checkpoint_sha256=e['checkpoint_sha256'],protocol_sha256=sha(R/'protocol.json'),updated_utc=datetime.datetime.utcnow().isoformat()+'Z')
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('BATCH_SIZE_COMPLETE',o.id,bs,flush=True)
print('BENCHMARK_COMPLETE',o.id,flush=True)
