#!/usr/bin/env python3
"""Materialize an archived run at a new path, without starting training."""
import argparse, ast, hashlib, json, os, shlex, shutil, sys
from pathlib import Path

RELEASE=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def parser_options(source):
 result={}
 for node in ast.walk(ast.parse(source)):
  if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='add_argument':continue
  flags=[a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value.startswith('--')]
  if not flags:continue
  kw={k.arg:k.value.value for k in node.keywords if isinstance(k.value,ast.Constant)}
  dest=kw.get('dest',flags[0][2:].replace('-','_'))
  result[dest]=(flags[0],kw.get('action'))
 return result
def replace_paths(value,mapping):
 if isinstance(value,dict):return {replace_paths(k,mapping):replace_paths(v,mapping) for k,v in value.items()}
 if isinstance(value,list):return [replace_paths(x,mapping) for x in value]
 if isinstance(value,str):
  for old,new in sorted(mapping.items(),key=lambda x:-len(x[0])):
   if value==old or value.startswith(old+'/'):return new+value[len(old):]
 return value
def prepare(args):
 registry=json.loads((RELEASE/'registry.json').read_text())
 record=next((r for r in registry['runs'] if r['id']==args.id),None)
 if record is None:raise ValueError('Unknown run ID: '+args.id)
 workspace=Path(args.workspace).resolve()
 if workspace.exists():raise FileExistsError('Use a new workspace; existing output is never replaced: '+str(workspace))
 if workspace in RELEASE.parents or workspace==RELEASE:raise ValueError('Workspace must not replace the release')
 assets=json.loads(Path(args.assets).read_text())
 for key in ['datasets_dir','bert_dir','cache_root']:
  assets[key]=str(Path(assets[key]).expanduser().resolve())
 assets['correlation_checkpoints']={k:str(Path(v).expanduser().resolve()) for k,v in assets['correlation_checkpoints'].items()}
 snapshot=registry['snapshots'][record['snapshot']]
 src=RELEASE/'source_snapshots'/record['snapshot']
 for name,digest in snapshot['files'].items():
  if sha(src/name)!=digest:raise ValueError('Archived source changed: '+name)
 cfg=json.loads((RELEASE/'runs'/args.id/'run_config.json').read_text())
 orig=cfg['arguments']; ds=record['dataset']
 policy=json.loads((src/'protocol/protocol.json').read_text())
 mapping={snapshot['original_root']:str(workspace),
  '/data01/lyw/GLoMo/datasets':assets['datasets_dir'],
  '/data01/lyw/GLoMo/BERT_EN':assets['bert_dir'],
  policy['cache_root']:assets['cache_root']}
 for key,old in policy['correlation_checkpoints'].items():mapping[old]=assets['correlation_checkpoints'][key]
 relocated=replace_paths(orig,mapping)
 relocated['experiment_root']=str(workspace/'runs')
 # The downstream encoder loads the neighboring BERT_EN; the cache uses this same checkpoint.
 relocated['corr_text_encoder_path']=str(workspace/'BERT_EN')
 options=parser_options((src/'Correlation_Guided_MSA/main_cgmsa.py').read_text())
 cli=[sys.executable,'main_cgmsa.py']
 for key,value in relocated.items():
  if key=='num_classes':continue  # Derived by the original parser.
  if key not in options:raise ValueError('Unrecognized archived option: '+key)
  flag,action=options[key]
  if action in ['store_true','store_false']:
   if value==(action=='store_true'):cli.append(flag)
  elif value is not None:cli += [flag]+[str(x) for x in (value if isinstance(value,list) else [value])]
 # Assert every non-path argument remains identical to the archived run.
 path_keys={'experiment_root','corr_model_path','corr_cache_root','corr_text_encoder_path'}
 assert all(relocated[k]==v for k,v in orig.items() if k not in path_keys)
 shutil.copytree(src,workspace)
 (workspace/'datasets').symlink_to(assets['datasets_dir'],target_is_directory=True)
 (workspace/'BERT_EN').symlink_to(assets['bert_dir'],target_is_directory=True)
 (workspace/'runs').mkdir()
 policy=replace_paths(policy,mapping)
 (workspace/'protocol/protocol.json').write_text(json.dumps(policy,indent=2)+'\n')
 inputs=json.loads((src/'protocol/input_manifest.json').read_text())
 needed_bert={'config.json','pytorch_model.bin','model.safetensors','vocab.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json'}
 selected=[]; omitted=[]
 guided=orig['use_fusion_correlation'] or orig['use_moe_reliability']
 for item in inputs['files']:
  old=item['path']; keep=item['dataset']==ds or item['dataset']=='shared'
  if '/BERT_EN/' in old:keep=Path(old).name in needed_bert and (Path(assets['bert_dir'])/Path(old).name).exists()
  if not guided and ('fixed_contextual_word_v1' in old or old.endswith('best_correlation.pt')):keep=False
  if keep:selected.append(replace_paths(item,mapping))
  else:omitted.append(old)
 if not any(Path(x['path']).name=='config.json' for x in selected):raise FileNotFoundError('BERT config.json is required')
 if not any(Path(x['path']).name in ['pytorch_model.bin','model.safetensors'] for x in selected):raise FileNotFoundError('BERT model weights are required')
 inputs['files']=selected
 (workspace/'protocol/input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
 frozen=json.loads((src/'protocol/frozen_source.json').read_text())
 frozen['files']={name:sha(workspace/name) for name in frozen['files']}
 (workspace/'protocol/frozen_source.json').write_text(json.dumps(frozen,indent=2)+'\n')
 env={'CUDA_VISIBLE_DEVICES':str(args.gpu),'PYTHONHASHSEED':str(record['seed']),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
 cmd={'cwd':str(workspace/'Correlation_Guided_MSA'),'argv':cli,'environment':env}
 (workspace/'launch.json').write_text(json.dumps(cmd,indent=2)+'\n')
 shell='#!/bin/sh\nset -eu\ncd '+shlex.quote(cmd['cwd'])+'\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'exec '+shlex.join(cli)+'\n'
 (workspace/'run.sh').write_text(shell);(workspace/'run.sh').chmod(0o755)
 report={'run':args.id,'snapshot':record['snapshot'],'path_mapping':mapping,'nonpath_arguments_unchanged':True,
  'input_checks_omitted':omitted,'reason':'Only assets consumed by this run are required; original complete input manifest remains in the immutable snapshot.',
  'checkpoint_sha256':record['checkpoint_sha256'],'started_training':False}
 (workspace/'relocation.json').write_text(json.dumps(report,indent=2)+'\n')
 if args.check_inputs:
  for item in selected:
   if sha(item['path'])!=item['sha256']:raise ValueError('Input hash differs from archived run: '+item['path'])
  report['input_hashes_verified']=len(selected)
  (workspace/'relocation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'workspace':str(workspace),'run':args.id,'command':str(workspace/'run.sh'),'training_started':False,'checked_inputs':args.check_inputs},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--id',required=True);p.add_argument('--assets',required=True);p.add_argument('--workspace',required=True)
 p.add_argument('--gpu',type=int,default=0);p.add_argument('--check-inputs',action='store_true')
 a=p.parse_args()
 if a.gpu<0:p.error('GPU index must be nonnegative')
 prepare(a)
