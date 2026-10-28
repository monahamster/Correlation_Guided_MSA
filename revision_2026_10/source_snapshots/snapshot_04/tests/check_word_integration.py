from pathlib import Path
import os,sys,json,subprocess,shlex,importlib.util
import torch,numpy as np
R=Path(__file__).resolve().parents[1];os.chdir(R/'Correlation_Guided_MSA');sys.path.insert(0,str(Path.cwd()));sys.argv=[sys.argv[0]]
os.environ['PYTHONHASHSEED']='5576';torch.set_num_threads(4)
cmd=subprocess.check_output([sys.executable,str(R/'scripts/run_formal.py'),'mosi','constant','5576','--gpu','0'],text=True)
sys.argv=shlex.split(cmd.splitlines()[1])[3:]
import main_cgmsa as main
from formal_protocol import tensor_digest
main.validate_formal_args()
train,dev,unused_test,unused_examples,steps=main.set_up_data_loader()
main.set_random_seed(5576);model=main.prep_for_training(steps);model.eval();before=tensor_digest(model.corr_model.state_dict().items())
C=Path(json.loads((R/'protocol/protocol.json').read_text())['correlation_candidate'])
spec=importlib.util.spec_from_file_location('candidate_readout',C/'readout.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
standalone=candidate.load_model(main.DEVICE)
batch=tuple(t.to(main.DEVICE) for t in next(iter(dev)));*_,corr=main.unpack_batch(batch)
model.correlation_control='none'
actual=model._correlation_signals(corr['corr_text'],corr['corr_audio'],corr['corr_visual'],corr['corr_mask'])
expected=candidate.score(standalone,corr['corr_text'],corr['corr_audio'],corr['corr_visual'],corr['corr_mask'].bool())
errors={}
for key,names in [('cosines',['raw_cosine_ta','raw_cosine_tv','raw_cosine_av']),('q',['q_t','q_a','q_v']),('scales',['s_t','s_a','s_v'])]:
 actual_tensor=torch.stack([actual[n] for n in names],1);errors[key]=float((actual_tensor-expected[key]).abs().max());assert torch.allclose(actual_tensor,expected[key],atol=2e-6)
model.correlation_control='constant';state=torch.get_rng_state().clone();gpu_state=torch.cuda.get_rng_state_all()
values=main.set_training_scale_means(model,train)
assert torch.equal(state,torch.get_rng_state());assert all(torch.equal(a,b) for a,b in zip(gpu_state,torch.cuda.get_rng_state_all()))
ref=json.loads((C/'validation.json').read_text());assert len(train.dataset)==ref['complete_train_samples']==1281
assert np.max(np.abs(np.array(values)-np.array(ref['train_scale_means'])))<2e-6
controlled=model._correlation_signals(corr['corr_text'],corr['corr_audio'],corr['corr_visual'],corr['corr_mask'])
for i,k in enumerate(['s_t','s_a','s_v']):assert torch.allclose(controlled[k],torch.full_like(controlled[k],values[i]),atol=1e-7)
assert tensor_digest(model.corr_model.state_dict().items())==before
report=dict(passed=True,real_dev_batch_size=len(batch[0]),integrated_vs_standalone_max_error=errors,complete_train_samples=len(train.dataset),constant_scales=values,constant_matches_candidate_full_train=True,constant_preserves_rng=True,correlation_weights_unchanged=True,constant_applied_to_guidance=True,test_evaluated=False)
(R/'protocol/word_integration_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
