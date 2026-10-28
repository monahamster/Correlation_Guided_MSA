from pathlib import Path
import subprocess,json,sys,hashlib
R=Path(__file__).resolve().parent;Q=R/'replications';arm=sys.argv[1]
for seed in [42,2026]:
 D=Q/('seed{}_{}'.format(seed,arm))
 for k,v in json.loads((D/'frozen_source.json').read_text()).items():assert hashlib.sha256((D/k).read_bytes()).hexdigest()==v
 with (D/'pipeline.log').open('x') as f:subprocess.run([sys.executable,str(D/'scripts/pilot.py'),'mosi','same_length'],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('COMPLETE',D.name,flush=True)
