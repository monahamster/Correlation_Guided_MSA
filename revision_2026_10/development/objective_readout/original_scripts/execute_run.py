from pathlib import Path
import sys,json,hashlib,subprocess
R=Path(__file__).resolve().parents[1];e=json.loads((R/'run_matrix.json').read_text())[int(sys.argv[1])];D=Path(e['root']);assert not e['reused']
for name,sha in json.loads((D/'frozen_source.json').read_text()).items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==sha
subprocess.run([sys.executable,str(D/'scripts/pilot.py'),'mosei','same_length'],cwd=D,check=True)
