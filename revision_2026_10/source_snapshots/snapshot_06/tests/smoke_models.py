"""Short synthetic forward/backward checks; no benchmark training or metrics."""
import os,sys,json,gc,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT/'Correlation_Guided_MSA')
sys.path.insert(0,str(ROOT/'Correlation_Guided_MSA'))
sys.argv=[sys.argv[0]]
import main_cgmsa as main
from formal_protocol import backbone_digest,seeded_loader,tensor_digest

torch.set_num_threads(4)
policy=json.loads((ROOT/'protocol/protocol.json').read_text())
results={}
for ds in ['mosei']:
    for k,v in policy['datasets'][ds].items():setattr(main.args,k,v)
    main.args.dataset=ds;main.args.seed=5576;main.args.corr_alpha=.2
    main.args.corr_model_path=policy['correlation_checkpoints'][ds]
    digest=None
    for variant in ['dynamic','constant']:
        guided=variant!='baseline'
        main.args.use_fusion_correlation=guided;main.args.use_moe_reliability=guided
        main.args.correlation_control='constant' if variant=='constant' else 'none'
        main.set_random_seed(5576)
        model=main.prep_for_training(2)
        current=backbone_digest(model)
        if digest is None:digest=current
        assert current==digest,(ds,variant,'backbone initialization mismatch')
        n=2;seq=8;words=5;torch.manual_seed(123)
        ids=torch.randint(100,200,(n,seq));visual=torch.randn(n,seq,main.args.VISUAL_DIM)
        audio=torch.randn(n,seq,74);mask=torch.ones_like(ids);seg=torch.zeros_like(ids);labels=torch.tensor([-.6,1.2])
        tensors=[ids,visual,audio,mask,seg,labels]
        if guided:
            tensors += [torch.randn(n,words,768),torch.randn(n,words,74),torch.randn(n,words,main.args.VISUAL_DIM),torch.ones(n,words,dtype=torch.bool)]
        loader=seeded_loader(TensorDataset(*tensors),2,True,100)
        old_rng=torch.get_rng_state().clone();old_cuda=torch.cuda.get_rng_state_all()
        constants=main.set_training_scale_means(model,loader)
        assert torch.equal(old_rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(old_cuda,torch.cuda.get_rng_state_all()))
        corr_before=tensor_digest(model.corr_model.state_dict().items()) if guided else None
        calls=[0]
        if guided:handle=model.corr_model.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
        opt=main.build_optimizer(model)
        main.set_random_seed(45576)
        losses=[main.train_epoch(model,loader,opt,epoch) for epoch in range(2)]
        assert all(torch.isfinite(torch.tensor(losses)))
        assert int(opt.state[model.regressor.weight]['step'])==2
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
        if guided:
            assert model.training and not model.corr_model.training
            assert not any(p.grad is not None for p in model.corr_model.parameters())
            assert tensor_digest(model.corr_model.state_dict().items())==corr_before
            assert calls[0]==2
            handle.remove()
        model.eval()
        pred1,_=main.test_epoch(model,seeded_loader(loader.dataset,2,False,20))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'state.pt';torch.save(model.state_dict(),path)
            if variant=='constant':model.constant_scales.zero_()
            model.load_state_dict(torch.load(path,map_location=main.DEVICE),strict=True)
        pred2,_=main.test_epoch(model,seeded_loader(loader.dataset,2,False,20))
        assert (pred1==pred2).all()
        results[ds+'_'+variant]={'shared_initialization_sha256':current,'two_step_losses':losses,'optimizer_step':2,'rng_unchanged_by_constants':True,'correlation_frozen':True if guided else None,'one_correlation_call_per_forward':True if guided else None,'reload_predictions_exact':True,'constant_scales':constants}
        print('PASSED',ds,variant,flush=True)
        del model,opt,loader;gc.collect();torch.cuda.empty_cache()
(ROOT/'protocol/model_validation.json').write_text(json.dumps(results,indent=2))
print('BOTH MOSEI MODEL CHECKS PASSED')
