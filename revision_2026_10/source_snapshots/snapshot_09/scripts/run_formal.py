#!/usr/bin/env python
"""Print a frozen single-run command; --execute starts that run."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('dataset',choices=['mosi'])
p.add_argument('variant',choices=['baseline','constant','dynamic'])
p.add_argument('seed',type=int,choices=[2027,3407])
p.add_argument('--gpu',required=True,type=int)
p.add_argument('--execute',action='store_true')
a=p.parse_args()
if a.gpu < 0: p.error('GPU index must be nonnegative')
config=json.loads((ROOT/'protocol/protocol.json').read_text())
tag='{}_{}_{}_seed{}'.format(config['version'],a.dataset,a.variant,a.seed)
cmd=[sys.executable,'main_cgmsa.py','--dataset',a.dataset,'--seed',str(a.seed),
     '--experiment_root',str(ROOT/'runs'),'--experiment_tag',tag]
for k,v in {**config['shared'],**config['datasets'][a.dataset]}.items():
    cmd+=['--'+k]+([str(x) for x in v] if isinstance(v,list) else [str(v)])
if a.variant!='baseline':
    cmd+=['--use_fusion_correlation','--use_moe_reliability',
          '--corr_model_path',config['correlation_checkpoints'][a.dataset],
          '--corr_cache_root',config['cache_root'],
          '--correlation_control','constant' if a.variant=='constant' else 'none']
print('cwd:',ROOT/'Correlation_Guided_MSA')
print('CUDA_VISIBLE_DEVICES={} PYTHONHASHSEED={} {}'.format(a.gpu,a.seed,shlex.join(cmd)),flush=True)
if a.execute:
    if (ROOT/'runs'/tag).exists(): raise SystemExit('Refusing existing run directory')
    logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONHASHSEED=str(a.seed),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (logdir/(tag+'.log')).open('x') as out:
        proc=subprocess.Popen(cmd,cwd=ROOT/'Correlation_Guided_MSA',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True);out.write(line);out.flush()
        raise SystemExit(proc.wait())
