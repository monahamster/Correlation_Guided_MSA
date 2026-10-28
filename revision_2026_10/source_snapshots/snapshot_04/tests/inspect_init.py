import os,sys,torch
from pathlib import Path
r=Path(__file__).resolve().parents[1];os.chdir(r/'Correlation_Guided_MSA');sys.path.insert(0,str(r/'Correlation_Guided_MSA'));sys.argv=[sys.argv[0]]
import main_cgmsa as m
torch.set_num_threads(4);m.args.VISUAL_DIM=47;m.args.layers=4
m.args.corr_model_path='/data01/lyw/GLoMo_repair_20261019T001/repair_runs/pretrain_mosi_seed5576_fixedseq_v2/best_correlation.pt'
states=[]
for enabled in [False,True]:
 m.args.use_fusion_correlation=enabled;m.args.use_moe_reliability=enabled;m.set_random_seed(5576)
 model=m.prep_for_training(1);states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('corr_model.') and not k.endswith('.w_rel') and k!='constant_scales'});del model
for k,v in states[0].items():
 if not torch.equal(v,states[1][k]): print('DIFF',k,tuple(v.shape),float((v-states[1][k]).abs().max()))
