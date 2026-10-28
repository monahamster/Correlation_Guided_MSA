from pathlib import Path
import os,sys,json,subprocess,shlex
D=Path(sys.argv[1]);os.chdir(D/'Correlation_Guided_MSA');sys.path.insert(0,str(Path.cwd()));sys.argv=[sys.argv[0]]
import main_cgmsa as m
checks=[]
for seed in [42,2026]:
 os.environ['PYTHONHASHSEED']=str(seed)
 for variant in ['dynamic','constant']:
  out=subprocess.check_output([sys.executable,str(D/'scripts/run_formal.py'),'mosei',variant,str(seed),'--gpu','0'],text=True)
  m.args=m.parser.parse_args(shlex.split(out.splitlines()[1])[4:]);m.apply_dataset_defaults(m.args);m.validate_formal_args()
  assert m.args.n_epochs==100 and m.args.learning_rate==1e-5 and m.args.train_batch_size==64
  assert m.args.use_fusion_correlation and m.args.use_moe_reliability
  checks.append([seed,variant]);m.args.save_best_by='acc2'
  try:m.validate_formal_args()
  except ValueError:pass
  else:raise AssertionError('Invalid checkpoint-selection criterion accepted')
assert not (D/'runs').exists()
(D/'protocol/launcher_validation.json').write_text(json.dumps(dict(passed=True,combinations=checks,selection_guard_passed=True,no_training_runs_created=True),indent=2)+'\n');print('PASSED',D.name)
