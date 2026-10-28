import os,sys,subprocess,shlex,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT/'Correlation_Guided_MSA')
sys.path.insert(0,str(ROOT/'Correlation_Guided_MSA'))
sys.argv=[sys.argv[0]]
import main_cgmsa as m
os.environ['PYTHONHASHSEED']='5576'
checks=[]
for ds in ['mosi']:
 for variant in ['baseline','constant','dynamic']:
  text=subprocess.check_output([sys.executable,str(ROOT/'scripts/run_formal.py'),ds,variant,'5576','--gpu','0'],text=True)
  tokens=shlex.split(text.splitlines()[1]);m.args=m.parser.parse_args(tokens[4:]);m.apply_dataset_defaults(m.args)
  m.validate_formal_args();checks.append(ds+'_'+variant)
  m.args.save_best_by='acc2'
  try:m.validate_formal_args()
  except ValueError:pass
  else:raise AssertionError('test-set selection was accepted')
assert not (ROOT/'runs').exists()
report={'three_launcher_configurations':checks,'test_set_selection_rejected':True,'dry_run_created_no_training_runs':True}
(ROOT/'protocol/launcher_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
