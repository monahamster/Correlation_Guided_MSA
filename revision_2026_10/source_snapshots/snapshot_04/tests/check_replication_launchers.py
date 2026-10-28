import os,sys,subprocess,shlex,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];os.chdir(R/'Correlation_Guided_MSA');sys.path.insert(0,str(R/'Correlation_Guided_MSA'));sys.argv=[sys.argv[0]]
import main_cgmsa as m
checks=[]
for seed in [42,2026]:
 os.environ['PYTHONHASHSEED']=str(seed)
 for variant in ['dynamic','constant']:
  out=subprocess.check_output([sys.executable,str(R/'scripts/run_formal.py'),'mosi',variant,str(seed),'--gpu','0'],text=True)
  tokens=shlex.split(out.splitlines()[1]);m.args=m.parser.parse_args(tokens[4:]);m.apply_dataset_defaults(m.args);m.validate_formal_args()
  assert m.args.n_epochs==70 and m.args.corr_model_path.endswith('word_sentence_triplet/runs/mosi_same_length/best_correlation.pt')
  checks.append([seed,variant]);m.args.save_best_by='acc2'
  try:m.validate_formal_args()
  except ValueError:pass
  else:raise AssertionError('invalid selection accepted')
assert not (R/'runs').exists()
report={'accepted_pairs':checks,'selection_guard_passed':True,'no_training_runs_created':True}
(R/'protocol/replication_launcher_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
