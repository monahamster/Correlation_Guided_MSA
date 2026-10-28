from pathlib import Path
import sys,subprocess,json,hashlib,ast
R=Path(__file__).resolve().parents[1];D=Path(sys.argv[1])
for test in ['check_dev_only.py','smoke_models.py']:
 with (D/'protocol'/(test+'.log')).open('w') as out:subprocess.run([sys.executable,str(D/'tests'/test)],cwd=D,stdout=out,stderr=subprocess.STDOUT,check=True)
with (D/'protocol/readout_validation.log').open('w') as out:subprocess.run([sys.executable,str(R/'scripts/check_readout.py'),str(D)],cwd=D,stdout=out,stderr=subprocess.STDOUT,check=True)
for f in D.rglob('*.py'):ast.parse(f.read_text())
paths=[p for d in ['Correlation_Guided_MSA','Correlation_Pretraining','scripts','tests'] for p in (D/d).rglob('*.py')]+[D/'protocol'/f for f in ['protocol.json','input_manifest.json','environment.json']]
manifest=dict(version=json.loads((D/'protocol/protocol.json').read_text())['version'],files={str(f.relative_to(D)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(paths)})
(D/'protocol/frozen_source.json').write_text(json.dumps(manifest,indent=2)+'\n');print('PREFLIGHT_PASSED',D.name,flush=True)
