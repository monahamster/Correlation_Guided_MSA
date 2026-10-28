from pathlib import Path
import os,sys,json,time,subprocess,datetime
R=Path(__file__).resolve().parents[1];entries=json.loads((R/'run_matrix.json').read_text());roots={e['readout']:Path(e['root']) for e in entries};jobs=json.loads((R/'queue.json').read_text());processes={}
def save(name,value):
 f=R/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(f)
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def directory(root,seed,variant):return root/'runs'/('{}_mosei_{}_seed{}'.format(read(root/'protocol/protocol.json')['version'],variant,seed))
def inspect(root,seed,variant,state):
 d=directory(root,seed,variant);history=[];h=d/'history.jsonl'
 if h.exists():
  for line in h.read_text().splitlines():
   try:history.append(json.loads(line))
   except json.JSONDecodeError:break
 m=read(d/'metrics.json');return dict(state='complete' if m else state,epochs_completed=len(history),best_valid_mae=min([x['valid_mae'] for x in history],default=None),result=m),read(d/'run_config.json')
try:
 assert not (R/'jobs.json').exists(),'Refusing controller restart that could duplicate jobs'
 while True:
  for i,proc in list(processes.items()):
   code=proc.poll()
   if code is not None:
    jobs[i]['exit_code']=code;jobs[i]['state']='complete' if code==0 and (directory(roots[jobs[i]['readout']],jobs[i]['seed'],jobs[i]['variant'])/'metrics.json').exists() else 'failed';del processes[i]
  if any(j['state']=='failed' for j in jobs):raise RuntimeError('Training failed; inspect launcher logs')
  active={jobs[i]['gpu'] for i in processes}
  if any(j['state']=='queued' for j in jobs):
   output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True);memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in output.splitlines()}
   for i,j in enumerate(jobs):
    if j['state']!='queued' or j['gpu'] in active or memory[j['gpu']]>=500:continue
    D=roots[j['readout']];cmd=[sys.executable,str(D/'scripts/run_formal.py'),'mosei',j['variant'],str(j['seed']),'--gpu',str(j['gpu']),'--execute']
    with (R/'logs'/('launcher_'+j['key']+'.log')).open('x') as out:proc=subprocess.Popen(cmd,cwd=D,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    processes[i]=proc;j.update(state='running',pid=proc.pid,command=cmd,launched_utc=datetime.datetime.utcnow().isoformat()+'Z');active.add(j['gpu']);print('STARTED',j['key'],proc.pid,flush=True)
  save('jobs.json',jobs);rows={};configs={}
  for j in jobs:
   rows[j['key']],c=inspect(roots[j['readout']],j['seed'],j['variant'],j['state'])
   if c:configs[j['key']]=(j['seed'],c)
  for e in entries:
   for v in ['dynamic','constant']:
    key='seed5576_'+e['readout']+'_'+v;rows[key],c=inspect(Path(e['pilot_root']),5576,v,'complete')
    if c:configs[key]=(5576,c)
  checks={};references={}
  for key,(seed,c) in configs.items():
   ref=references.setdefault(seed,c);check={k:c[k]==ref[k] for k in ['backbone_initialization_sha256','routing_initialization_sha256','samples','input_sha256']};assert all(check.values()),(key,check);checks[key]=check
  save('paired_initialization.json',checks)
  state='summarizing' if all(j['state']=='complete' for j in jobs) else 'running';save('STATUS.json',dict(state=state,runs=rows,initialization_checked=list(checks),queued=sum(j['state']=='queued' for j in jobs),running=len(processes)))
  if state=='summarizing':break
  time.sleep(30)
 with (R/'logs/comparison.log').open('w') as out:subprocess.run([sys.executable,str(R/'scripts/summarize.py')],stdout=out,stderr=subprocess.STDOUT,check=True)
 save('STATUS.json',dict(state='complete',runs=rows,initialization_checked=list(checks),comparison=str(R/'comparison.json')));print('complete',flush=True)
except Exception as exc:
 save('jobs.json',jobs);save('STATUS.json',dict(state='failed',error=repr(exc),runs=locals().get('rows',{})));raise
