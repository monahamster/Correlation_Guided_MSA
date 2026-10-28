from pathlib import Path
import sys,json,time,subprocess,datetime,hashlib,fcntl,os
R=Path(__file__).resolve().parents[1]
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def now():return datetime.datetime.utcnow().isoformat()+'Z'
frozen=json.loads((R/'execution_manifest.json').read_text())
def verify():
    for name,digest in frozen.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
jobs=[]
for e in json.loads((R/'evaluation_manifest.json').read_text())['entries']:
    jobs.append(dict(id=e['id'],kind='model',status='queued',command=[sys.executable,str(R/'scripts/benchmark.py'),'--id',e['id'],'--gpu','0']))
for ds in ['mosi','mosei']:
    jobs.append(dict(id='cache_'+ds,kind='cache',status='queued',command=[sys.executable,str(R/'scripts/cache_benchmark.py'),'--dataset',ds,'--gpu','0']))
assert not (R/'jobs.json').exists()
for j in jobs:
    verify()
    while True:
        raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip()
        mem,util=map(int,raw.split(','))
        if mem<500 and util<5:break
        save(R/'STATUS.json',dict(status='waiting_for_idle_gpu',next=j['id'],updated_utc=now()));time.sleep(10)
    with (R/'logs'/(j['id']+'.log')).open('x') as log:
        proc=subprocess.Popen(j['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        j.update(status='running',pid=proc.pid,started_utc=now());save(R/'jobs.json',jobs)
        save(R/'STATUS.json',dict(status='running',current=j['id'],completed=sum(x['status']=='complete' for x in jobs),total=len(jobs),controller_pid=os.getpid(),updated_utc=now()))
        code=proc.wait()
    j.update(status='complete' if code==0 else 'failed',exit_code=code,finished_utc=now());save(R/'jobs.json',jobs)
    if code:
        save(R/'STATUS.json',dict(status='failed',current=j['id'],exit_code=code,updated_utc=now()));sys.exit(code)
verify();save(R/'STATUS.json',dict(status='measurements_complete',completed=8,total=8,updated_utc=now()))
print('EFFICIENCY_MEASUREMENTS_COMPLETE',flush=True)
