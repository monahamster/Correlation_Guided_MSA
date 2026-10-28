from pathlib import Path
import json,subprocess,sys,time,datetime
R=Path(__file__).resolve().parents[1]
while True:
    s=json.loads((R/'STATUS.json').read_text())
    if s['status']=='complete':break
    if s['status'] in ['failed','summary_failed']:raise RuntimeError(s)
    time.sleep(10)
for name in ['validate_results.py','plot_results.py']:
    with (R/'logs'/(name+'.log')).open('x') as log:
        subprocess.check_call([sys.executable,str(R/'scripts'/name)],stdout=log,stderr=subprocess.STDOUT)
(R/'postprocess_status.json').write_text(json.dumps(dict(status='complete',validation='validation.json',figures='figures',completed_utc=datetime.datetime.utcnow().isoformat()+'Z'),indent=2))
