from pathlib import Path
import json,tempfile,subprocess,time
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];source=(R/'scripts/controller.py').read_text();jobs=json.loads((R/'queue.json').read_text());clock=[0];procs=[];launches=[];peaks=[]
with tempfile.TemporaryDirectory(dir=R) as td:
 T=Path(td);(T/'logs').mkdir();(T/'scripts').mkdir();entries=[]
 for readout in ['sentence','word']:
  D=T/readout;(D/'protocol').mkdir(parents=True);(D/'protocol/protocol.json').write_text(json.dumps({'version':readout}));entries.append(dict(readout=readout,root=str(D),pilot_root=str(D)))
 (T/'run_matrix.json').write_text(json.dumps(entries));(T/'queue.json').write_text(json.dumps(jobs))
 class Proc:
  def __init__(self,cmd,**kw):
   D=Path(kw['cwd']);seed=int(cmd[4]);variant=cmd[3];gpu=int(cmd[6]);self.done=False;self.end=clock[0]+2;self.pid=10000+len(procs);self.dir=D/'runs'/('{}_mosei_{}_seed{}'.format(D.name,variant,seed));self.dir.mkdir(parents=True);self.gpu=gpu
   assert not any(p.gpu==gpu and not p.done for p in procs),'GPU overlap'
   assert not (clock[0]==0 and gpu==0),'Occupied GPU was used'
   launches.append((seed,D.name,variant,gpu,clock[0]));procs.append(self);peaks.append(sum(not p.done for p in procs))
  def poll(self):
   if clock[0]>=self.end:
    self.done=True;(self.dir/'metrics.json').write_text('{}');return 0
   return None
 def gpu(*a,**kw):return '\n'.join('{}, {}'.format(i,2000 if (i in [1,7] or (clock[0]==0 and i==0)) else 6) for i in range(8))
 def sleep(*a):clock[0]+=1;assert clock[0]<15,'Queue stalled'
 with patch.object(subprocess,'Popen',Proc),patch.object(subprocess,'check_output',gpu),patch.object(subprocess,'run',return_value=type('Result',(),{'returncode':0})()),patch.object(time,'sleep',sleep):
  exec(compile(source,'controller-under-test','exec'),{'__file__':str(T/'scripts/controller.py'),'__name__':'__main__'})
 final=json.loads((T/'jobs.json').read_text());assert len(launches)==8 and all(j['state']=='complete' for j in final);assert max(peaks)<=6
 for g in [0,2]:
  group=[x for x in launches if x[3]==g];assert len(group)==2 and group[1][4]>=group[0][4]+2
 report=dict(passed=True,all_eight_jobs_started_once=True,max_simultaneous=max(peaks),occupied_gpu_skipped=True,queued_gpu_reuse_waited_for_completion=True,launch_order=launches)
(R/'controller_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
