from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parent;arm=sys.argv[1]
for seed in [5576,42,2026]:
 name='{}_seed{}'.format(arm,seed)
 with (R/name/'pipeline.log').open('x') as f:subprocess.run([sys.executable,str(R/'execute_run.py'),name],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'COMPLETE',flush=True)
