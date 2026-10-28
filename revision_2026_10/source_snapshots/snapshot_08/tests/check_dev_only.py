from pathlib import Path
import os,sys,json,tempfile,subprocess,shlex
from unittest.mock import patch
import torch,numpy as np
R=Path(__file__).resolve().parents[1];os.chdir(R/'Correlation_Guided_MSA');sys.path.insert(0,str(Path.cwd()));sys.argv=[sys.argv[0]]
import main_cgmsa as m
os.environ['PYTHONHASHSEED']='5576'
for variant in ['dynamic','constant']:
 text=subprocess.check_output([sys.executable,str(R/'scripts/run_formal.py'),'mosei',variant,'5576','--gpu','0'],text=True)
 m.args=m.parser.parse_args(shlex.split(text.splitlines()[1])[4:]);m.apply_dataset_defaults(m.args);m.validate_formal_args()
model=torch.nn.Linear(1,1).to(m.DEVICE);train=object();dev=object();test=object();seen=[]
def evaluate(model,loader):
 assert loader is dev,'Attempted to evaluate test';seen.append('dev');return np.array([0.,1.]),np.array([.2,.8])
with tempfile.TemporaryDirectory() as td:
 p=Path(td);m.args.n_epochs=3
 with patch.object(m,'build_optimizer',return_value=type('DummyOptimizer',(),{'param_groups':[{'lr':1e-5}]})()),patch.object(m,'train_epoch',return_value=.5),patch.object(m,'eval_epoch',side_effect=[(.25,.5),(.04,.2),(.09,.3)]),patch.object(m,'test_epoch',side_effect=evaluate),patch.object(m,'test_score_model',side_effect=AssertionError('Test score forbidden')),patch.object(m,'resolve_best_model_path',return_value=p/'best.pt'),patch.object(m,'resolve_experiment_dir',return_value=p),patch.object(m,'write_metrics_summary'):
  summary=m.train(model,train,dev,test)
  assert summary['best_epoch']==1 and summary['test_evaluated'] is False
  assert len(seen)==1 and (p/'dev_predictions.npz').exists()
(R/'protocol/dev_only_validation.json').write_text(json.dumps(dict(passed=True,launcher_variants=['dynamic','constant'],selected_dev_only=True,test_score_never_called=True,earliest_minimum_selection=True),indent=2)+'\n')
print('DEV_ONLY_VALIDATION_PASSED')
