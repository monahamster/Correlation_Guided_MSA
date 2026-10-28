from pathlib import Path
import json,os,sys,subprocess,time,datetime,fcntl,hashlib
R=Path(__file__).resolve().parents[1]
lock=(R/'controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.datetime.utcnow().isoformat()+'Z'
def save(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(path)
frozen=json.loads((R/'execution_manifest.json').read_text())
def verify():
    for name,digest in frozen.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
verify()
assert json.loads((R/'preflight_validation.json').read_text())['passed']
assert not (R/'jobs.json').exists(),'Existing queue requires inspection before restart'
es=json.loads((R/'evaluation_manifest.json').read_text())['entries']
es=sorted(es,key=lambda e:(e['seed'],e['id'].split('_')[1],e['dataset']))
jobs=[dict(id=e['id'],status='queued') for e in es]
running={};gpus=[0,2,3,4,5,6]
def sync():
    completed=sum(j['status']=='complete' for j in jobs);failed=sum(j['status']=='failed' for j in jobs)
    done=sum(json.loads((R/'evaluations'/j['id']/'progress.json').read_text())['completed'] for j in jobs if (R/'evaluations'/j['id']/'progress.json').exists())
    save(R/'jobs.json',jobs)
    save(R/'STATUS.json',dict(status='running' if running or any(j['status']=='queued' for j in jobs) else ('failed' if failed else 'evaluation_complete'),total=30,completed=completed,failed=failed,running=len(running),queued=sum(j['status']=='queued' for j in jobs),conditions_completed=done,conditions_total=570,controller_pid=os.getpid(),updated_utc=now()))
while any(j['status'] in ['queued','running'] for j in jobs):
    verify()
    for gpu,(proc,j,log) in list(running.items()):
        code=proc.poll()
        if code is not None:
            log.close();p=R/'evaluations'/j['id']/'progress.json'
            complete=p.exists() and json.loads(p.read_text())['status']=='complete'
            j.update(status='complete' if code==0 and complete else 'failed',exit_code=code,finished_utc=now())
            del running[gpu]
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode()
    mem={int(x.split(',')[0]):(int(x.split(',')[1]),int(x.split(',')[2])) for x in raw.strip().splitlines()}
    for gpu in gpus:
        if gpu in running or mem[gpu][0]>500 or mem[gpu][1]>5:continue
        j=next((x for x in jobs if x['status']=='queued'),None)
        if j is None:break
        if (R/'evaluations'/j['id']).exists():
            j.update(status='failed',reason='Existing output; inspection required');continue
        ps=subprocess.check_output(['ps','-eo','args']).decode().splitlines()
        assert not any(str(R/'scripts/evaluate.py') in line and '--id '+j['id']+' ' in line for line in ps),'Duplicate live evaluation'
        log=(R/'logs'/('test_'+j['id']+'.log')).open('x')
        proc=subprocess.Popen([sys.executable,'-u',str(R/'scripts/evaluate.py'),'--id',j['id'],'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        j.update(status='running',gpu=gpu,pid=proc.pid,started_utc=now())
        running[gpu]=(proc,j,log)
    sync()
    if running or any(j['status']=='queued' for j in jobs):time.sleep(10)
verify();sync()
if all(j['status']=='complete' for j in jobs):
    with (R/'logs/summary.log').open('x') as log:
        code=subprocess.call([sys.executable,str(R/'scripts/summarize.py')],stdout=log,stderr=subprocess.STDOUT)
    status=json.loads((R/'STATUS.json').read_text());status.update(status='complete' if code==0 else 'summary_failed',summary_exit_code=code,updated_utc=now());save(R/'STATUS.json',status)
    if code==0:
        p=Path('/data01/lyw/GLoMo/revision_plan_20261020/STATUS.json');central=json.loads(p.read_text())
        central.update(robustness_status='complete',robustness_results=str(R/'comparison.json'),robustness_completed_utc=now())
        save(p,central)
print('CONTROLLER_FINISHED',flush=True)
