from pathlib import Path
import os,sys,json,subprocess,threading,concurrent.futures,datetime,time,fcntl,hashlib
R=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.utcnow().isoformat()+'Z'
manifest=json.loads((R/'evaluation_manifest.json').read_text());entries=manifest['entries']
assert len(entries)==78 and json.loads((R/'extension_inventory_validation.json').read_text())['passed']
assert json.loads((R/'preflight_validation.json').read_text())['passed']
lockfile=(R/'test_controller.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'test_jobs.json').exists(),'Existing test queue; inspect instead of relaunching'
for e in entries:assert not (R/'evaluations/test'/e['id']).exists(),e['id']
gpus=[0,2,3,4,5,6];lock=threading.Lock();stop=threading.Event()
jobs={e['id']:dict(id=e['id'],dataset=e['dataset'],seed=e['seed'],state='queued') for e in entries}
logs=R/'test_logs';logs.mkdir(exist_ok=False)
def save():
 for name,data in [('test_jobs.json',jobs),('TEST_STATUS.json',dict(state='failed' if stop.is_set() else ('complete' if all(x['state']=='complete' for x in jobs.values()) else 'running'),updated_utc=now(),total=len(jobs),counts={s:sum(x['state']==s for x in jobs.values()) for s in ['queued','running','complete','failed']},controller_pid=os.getpid()))]:
  p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def available(gpu):
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
 return {int(s.split(',')[0]):int(s.split(',')[1]) for s in out.splitlines()}.get(gpu,999999)<500
def worker(gpu,items):
 for e in items:
  if stop.is_set():return
  while not available(gpu):
   if stop.is_set():return
   time.sleep(10)
  key=e['id']
  with lock:
   jobs[key].update(state='running',gpu=gpu,started_utc=now());save()
  command=[sys.executable,'-u',str(R/'scripts/evaluate_one.py'),'--id',key,'--split','test','--gpu',str(gpu)]
  try:
   with (logs/(key+'.log')).open('x') as f:
    proc=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    with lock:jobs[key]['pid']=proc.pid;save()
    code=proc.wait()
   if code:raise RuntimeError('Evaluator exit code '+str(code))
   out=R/'evaluations/test'/key;metrics=json.loads((out/'metrics.json').read_text())
   assert metrics['test_evaluated'] and metrics['checkpoint_sha256']==e['checkpoint_sha256'] and metrics['metrics']['n']==e['samples']['test']
   digest=hashlib.sha256((out/'predictions.npz').read_bytes()).hexdigest()
   with lock:jobs[key].update(state='complete',exit_code=code,finished_utc=now(),predictions_sha256=digest);save()
   print('TEST_COMPLETE',key,flush=True)
  except Exception as exc:
   stop.set()
   with lock:jobs[key].update(state='failed',error=repr(exc),finished_utc=now());save()
   print('TEST_FAILED',key,repr(exc),flush=True);return
save()
with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
 futures=[pool.submit(worker,g,entries[i::len(gpus)]) for i,g in enumerate(gpus)]
 for f in futures:f.result()
if not stop.is_set():
 assert all(j['state']=='complete' for j in jobs.values())
 with (R/'test_summary.log').open('x') as f:subprocess.run([sys.executable,str(R/'scripts/summarize_test.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('ALL_TEST_EVALUATIONS_COMPLETE',flush=True)
