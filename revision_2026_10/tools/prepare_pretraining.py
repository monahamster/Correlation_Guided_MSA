#!/usr/bin/env python3
"""Prepare the selected pretraining recipe without running an experiment."""
import argparse,json,shutil,hashlib,shlex,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['mosi','mosei'],required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
 if a.workspace.exists():p.error('Use a new workspace')
 if a.gpu<0:p.error('GPU index must be nonnegative')
 src=R/'pretraining'/a.dataset;out=a.workspace.resolve();out.mkdir(parents=True)
 shutil.copytree(src/'scripts',out/'scripts');shutil.copy2(src/'protocol.json',out/'protocol.json');(out/'runs').mkdir()
 old_cache='/data01/lyw/GLoMo_repair_20261019T001/repair_cache'
 old_manifest='/data01/lyw/GLoMo_formal_20261020/protocol/input_manifest.json'
 m=json.loads((R/'pretraining/input_manifest.json').read_text())
 # Keep the historical expected cache hashes. A different cache must not silently pass as the original input.
 m['files']=[{**x,'path':str(a.cache_root.resolve())+x['path'][len(old_cache):]} for x in m['files'] if x['path'].startswith(old_cache+'/') and x['dataset']==a.dataset]
 (out/'input_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 script=out/'scripts/pilot.py';s=script.read_text()
 assert old_cache in s and old_manifest in s
 s=s.replace(old_cache,str(a.cache_root.resolve())).replace(old_manifest,str(out/'input_manifest.json'));script.write_text(s)
 cmd=[sys.executable,str(script),a.dataset,'same_length']
 (out/'run.sh').write_text('#!/bin/sh\nset -eu\nexport CUDA_VISIBLE_DEVICES='+str(a.gpu)+'\nexport PYTHONHASHSEED=5576\nexec '+shlex.join(cmd)+'\n')
 (out/'run.sh').chmod(0o755)
 (out/'relocation.json').write_text(json.dumps({'dataset':a.dataset,'pretraining_seed':5576,'changes':'Paths only; original objective, masks, sampling, 20 epochs and checkpoint criterion preserved.','original_checkpoint':json.loads((src/'checkpoint.json').read_text()),'training_started':False},indent=2)+'\n')
 print('Prepared',out/'run.sh','; training has not started.')
if __name__=='__main__':main()
