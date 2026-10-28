from pathlib import Path
import json,datetime,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'evaluation_manifest.json').read_text())
rows=[];checks=[];raws={}
for e in manifest['entries']:
    v=json.loads((R/'evaluations'/e['id']/'result.json').read_text());assert v['status']=='complete'
    assert v['checkpoint_sha256']==e['checkpoint_sha256']
    assert v['protocol_sha256']==hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()
    assert v['validation']['max_instrumentation_error']<1e-5
    raws[e['id']]=v
    for m in v['measurements']:
        bs=m['batch_size'];assert m['latency']['n_calls']==150
        hook_dense=m['flop_details'][0]['total_flops']
        dl=json.loads((Path(e['run'])/'run_config.json').read_text())['arguments']['d_l']
        final_fusion=2*bs*dl*3
        dense=hook_dense+final_fusion
        native=sum(x['flops'] for x in m['native_profiler'] if x['operator'] in ['aten::mm','aten::bmm','aten::addmm','aten::conv2d'])
        conv=sum(x['flops'] for x in m['flop_details'][0]['operators'] if x['kind']=='Conv1d')
        fused_ff=0;fused_attn=0
        for x in m['native_profiler']:
            if x['operator']=='aten::_addmm_activation':
                a,b=x['input_shapes'][1:3];fused_ff+=x['count']*2*a[0]*a[1]*b[1]
            if x['operator'] in ['aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_flash_attention']:
                q,k,vv=x['input_shapes'][:3]
                fused_attn+=x['count']*2*q[0]*q[1]*q[2]*k[2]*(q[3]+vv[3])
        augmented=native+conv+fused_ff+fused_attn
        diff=dense-augmented
        checks.append(dict(id=e['id'],batch_size=bs,analytical=dense,hook_dense=hook_dense,functional_final_fusion_flops=final_fusion,profiler_dense=native,conv1d_supplement=conv,fused_ffn_supplement=fused_ff,fused_attention_supplement=fused_attn,residual=diff,relative_residual=abs(diff)/dense))
        assert abs(diff)/dense<1e-5,(e['id'],bs,dense,augmented)
        assert all(x['total_flops']==hook_dense for x in m['flop_details'])
        variant=e['id'].split('_')[1]
        rows.append(dict(dataset=e['dataset'],variant=variant,batch_size=bs,parameters=v['parameters']['total'],trainable_parameters=v['parameters']['trainable'],correlation_parameters=v['parameters']['correlation'],gflops_per_sample=(m['dense_flops_per_sample']+final_fusion/bs)/1e9,median_ms=m['latency']['median_wall_ms'],p95_ms=m['latency']['p95_wall_ms'],median_cuda_ms=m['latency']['median_cuda_ms'],throughput_samples_per_second=m['latency']['samples_per_second'],peak_allocated_mib=m['peak_allocated_bytes']/2**20,peak_reserved_mib=m['peak_reserved_bytes']/2**20,correlation_median_ms=m['correlation_latency']['median_wall_ms'] if m['correlation_latency'] else 0,seq_length=m['sequence_length'],corr_padded_words=m['correlation_padded_words']))
overhead=[]
for ds in ['mosi','mosei']:
    b=raws[ds+'_baseline_seed5576']['parameters'];d=raws[ds+'_dynamic_seed5576']['parameters'];c=raws[ds+'_constant_seed5576']['parameters']
    assert d['total']==c['total'] and d['trainable']==c['trainable']
    assert d['trainable']-b['trainable']==9
    assert d['total']-b['total']==d['correlation']+9
    for bs in [1,32]:
        base=next(x for x in rows if (x['dataset'],x['variant'],x['batch_size'])==(ds,'baseline',bs))
        dyn=next(x for x in rows if (x['dataset'],x['variant'],x['batch_size'])==(ds,'dynamic',bs))
        overhead.append(dict(dataset=ds,batch_size=bs,extra_parameters=d['total']-b['total'],extra_trainable_parameters=9,parameter_increase_percent=100*(d['total']/b['total']-1),gflops_increase_percent=100*(dyn['gflops_per_sample']/base['gflops_per_sample']-1),latency_increase_percent=100*(dyn['median_ms']/base['median_ms']-1),extra_peak_allocated_mib=dyn['peak_allocated_mib']-base['peak_allocated_mib']))
caches=[]
for ds in ['mosi','mosei']:
    c=json.loads((R/('cache_'+ds+'.json')).read_text());assert c['status']=='complete'
    assert all(x['ids_word_maps_audio_visual_exact'] for x in c['splits'])
    generator_dense_full_length=12*(24*c['seq_length']*768**2+4*c['seq_length']**2*768)+2*768**2
    for row in rows:
        if row['dataset']==ds:
            row['offline_generator_parameters']=c['generator_parameters'] if row['variant']!='baseline' else 0
            row['full_pipeline_parameter_inventory']=row['parameters']+row['offline_generator_parameters']
    caches.append(dict(dataset=ds,generator_dense_gflops_at_max_seq_length=generator_dense_full_length/1e9,generator_parameters=c['generator_parameters'],total_wall_seconds=sum(x['wall_seconds'] for x in c['splits']),total_cache_gib=sum(x['cache_file_bytes'] for x in c['splits'])/2**30,samples=sum(x['samples'] for x in c['splits']),max_gpu_allocated_mib=max(x['peak_gpu_allocated_bytes'] for x in c['splits'])/2**20,max_text_replay_error=max(x['max_text_replay_error'] for x in c['splits']),splits=c['splits']))
for name,digest in json.loads((R/'execution_manifest.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
result=dict(status='complete_verified',created_utc=datetime.datetime.utcnow().isoformat()+'Z',rows=rows,dynamic_overhead=overhead,cache_generation=caches,dense_flops_validation=checks,measurement='FP32 tensors, original TF32 flags recorded in raw results; RTX3090 GPU0, serial, cached resident inputs; seed5576 weights; 150 timing repetitions per batch size. FLOPs are declared dense arithmetic with multiply-add=2.',max_prediction_replay_error=max(v['validation']['max_saved_dev_replay_error'] for v in raws.values()),max_instrumentation_error=max(v['validation']['max_instrumentation_error'] for v in raws.values()))
(R/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
lines=['Efficiency: median/p95 latency in ms per batch; memory in MiB; dense GFLOPs per sample.']
for x in rows:
    lines.append('%s %s batch%d: %.3f M params, %.4f GFLOPs, %.3f / %.3f ms, %.1f MiB peak allocated'%(x['dataset'],x['variant'],x['batch_size'],x['parameters']/1e6,x['gflops_per_sample'],x['median_ms'],x['p95_ms'],x['peak_allocated_mib']))
for c in caches:lines.append('%s cache: %.3f s; %.4f GiB; %d samples'%(c['dataset'],c['total_wall_seconds'],c['total_cache_gib'],c['samples']))
(R/'results.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
