from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;S=Path('/data01/lyw/GLoMo_mosi_pretrain_opt_20261020')
arms=['reference','dropout_03','weight_decay_1e3'];pairs=['text_audio','text_visual','audio_visual'];seeds=[5576,42,2026];rows=[]
for seed in seeds:
 for arm in arms:
  d=(S/'word_sentence_triplet' if seed==5576 else S/'replications'/('seed{}_word_sentence'.format(seed))) if arm=='reference' else R/('{}_seed{}'.format(arm,seed))
  for k,h in json.loads((d/'frozen_source.json').read_text()).items():assert hashlib.sha256((d/k).read_bytes()).hexdigest()==h
  if arm!='reference':assert json.loads((d/'COMPLETE.json').read_text())['complete']
  s=json.loads((d/'runs/mosi_same_length/summary.json').read_text());v=s['dev']['same_length']
  row=dict(seed=seed,arm=arm,best_epoch=s['best_epoch'],dev_loss=v['loss'],auc={k:v['pairs'][k]['auc_mean'] for k in pairs},root=str(d))
  f=d/'train_dev_summary.json'
  if f.exists():
   row['train_dev_eval']=json.loads(f.read_text())['rows']
  rows.append(row)
stats={}
for arm in arms:
 a=[r for r in rows if r['arm']==arm];stats[arm]=dict(dev_loss_mean=float(np.mean([x['dev_loss'] for x in a])),dev_loss_sample_std=float(np.std([x['dev_loss'] for x in a],ddof=1)),auc_mean={k:float(np.mean([x['auc'][k] for x in a])) for k in pairs},auc_sample_std={k:float(np.std([x['auc'][k] for x in a],ddof=1)) for k in pairs})
deltas=[]
for seed in seeds:
 a=next(r for r in rows if r['seed']==seed and r['arm']=='reference')
 for arm in arms[1:]:
  b=next(r for r in rows if r['seed']==seed and r['arm']==arm);deltas.append(dict(seed=seed,arm=arm,loss_delta=b['dev_loss']-a['dev_loss'],auc_delta={k:b['auc'][k]-a['auc'][k] for k in pairs}))
winner=min(arms,key=lambda a:stats[a]['dev_loss_mean'])
report=dict(scope='MOSI development experiments: three training seeds, each evaluated on five fixed donor maps. Same dev set is used for model and candidate selection. No test results.',rows=rows,statistics=stats,paired_deltas=deltas,lowest_mean_validation_loss=winner)
(R/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(10.8,4.5),layout='constrained');labels=['Reference','Dropout 0.3','Adam L2 0.001'];cols=['#2c7bb6','#e67e22','#2ca078']
for seed,color in zip(seeds,cols):
 axs[0].plot(range(3),[next(r['dev_loss'] for r in rows if r['seed']==seed and r['arm']==a) for a in arms],'o-',label='Seed '+str(seed),color=color)
axs[0].set(xticks=range(3),xticklabels=labels,ylabel='Selected sentence-level validation loss',title='Paired regularization comparison');axs[0].legend(fontsize=8)
for i,(arm,col,label) in enumerate(zip(arms,cols,labels)):
 x=np.arange(3)+(i-1)*.11;axs[1].errorbar(x,[stats[arm]['auc_mean'][k] for k in pairs],yerr=[stats[arm]['auc_sample_std'][k] for k in pairs],fmt='o',capsize=3,color=col,label=label)
axs[1].axhline(.5,color='gray',linestyle='--',lw=1);axs[1].set(xticks=range(3),xticklabels=['Text-Audio','Text-Visual','Audio-Visual'],ylabel='Validation pairing AUC',title='Mean +/- sample SD over three training seeds');axs[1].legend(fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.2)
fig.suptitle('MOSI word + sentence triplet pretraining | fixed 20-epoch budget',fontsize=12)
for ext in ['png','svg']:fig.savefig(R/('regularization_comparison.'+ext),dpi=145)
print(json.dumps({'statistics':stats,'paired_deltas':deltas,'lowest_mean_validation_loss':winner},indent=2))
