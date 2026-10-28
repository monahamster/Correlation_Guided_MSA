import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data01/lyw/GLoMo_position_ablation_20261020')
D=Path('/data01/lyw/GLoMo_mosi_depth_probe_20261020')
B=Path('/data01/lyw/GLoMo_length_pretrain_20261020')
pairs=['text_audio','text_visual','audio_visual']
rows=[]
for ds in ['mosi','mosei']:
 for x in json.loads((R/'generalization'/ds/'summary.json').read_text())['rows']:
  for split,s in x['splits'].items():
   v=s['pairing']['same_length']; rows.append(dict(dataset=ds,configuration=x['position']+'_3layers',checkpoint=x['checkpoint'],epoch=x['epoch'],split=split,n=s['n'],loss=v['loss'],auc={p:v['pairs'][p]['auc_mean'] for p in pairs}))
for x in json.loads((D/'train_dev_summary.json').read_text())['rows']:
 for split,s in x['splits'].items():
  v=s['same_length']; rows.append(dict(dataset='mosi',configuration='without_position_1layer',checkpoint=x['checkpoint'],epoch=x['epoch'],split=split,n=1273 if split=='train' else 222,loss=v['loss'],auc={p:v['pairs'][p]['auc_mean'] for p in pairs}))
summary=dict(scope='Single training seed 5576; means over five fixed negative maps, not five training seeds. No test evaluation. Best checkpoints selected by same-length validation triplet loss.',rows=rows)
(R/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
configs=[('with_position_3layers','Position, 3 layers','#3072b5',B),('without_position_3layers','No position, 3 layers','#db7d31',R),('without_position_1layer','No position, 1 layer','#35966f',D)]
fig,axs=plt.subplots(1,2,figsize=(10.4,4.6),layout='constrained')
for ax,ds in zip(axs,['mosi','mosei']):
 for conf,label,col,root in configs:
  if ds=='mosei' and root==D: continue
  a=next(x for x in rows if x['dataset']==ds and x['configuration']==conf and x['checkpoint']=='best' and x['split']=='dev')
  ax.plot(range(3),[a['auc'][p] for p in pairs],'o-',label=label,color=col,linewidth=1.7)
 ax.axhline(.5,color='gray',linestyle='--',linewidth=1)
 ax.set(xticks=range(3),xticklabels=['Text-Audio','Text-Visual','Audio-Visual'],ylim=(.45,.78),title=ds.upper()+' | selected checkpoint',ylabel='Same-length pairing AUC')
 ax.grid(axis='y',alpha=.15)
axs[0].legend(fontsize=8,loc='upper left')
fig.suptitle('Validation pairing: isolated position and depth probes\nOne training seed; five fixed negative maps; no test evaluation',fontsize=11)
for ext in ['png','svg']: fig.savefig(R/('validation_pairing.'+ext),dpi=150)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.4,4.6),layout='constrained')
for conf,label,col,root in configs:
 h=json.loads((root/'runs/mosi_same_length/history.json').read_text())
 axs[0].plot([x['epoch'] for x in h],[x['train_loss'] for x in h],color=col,label=label)
 axs[1].plot([x['epoch'] for x in h],[x['dev']['same_length']['loss'] for x in h],color=col,label=label)
for ax,title in zip(axs,['Training (dropout on)','Validation (dropout off)']):
 ax.axhline(.4,color='gray',linestyle='--',linewidth=1,label='Equal positive/negative similarity')
 ax.set(title=title,xlabel='Epoch',ylabel='Triplet loss',xticks=[1,5,10,15,20]); ax.grid(alpha=.15)
axs[0].legend(fontsize=8)
fig.suptitle('MOSI: removing position enables fitting, but validation does not improve\nTraining and validation differ in dropout and evaluation donors',fontsize=11)
for ext in ['png','svg']: fig.savefig(R/('mosi_learning_curves.'+ext),dpi=150)
plt.close(fig)
print('SUMMARY_AND_PLOTS_COMPLETE')
for x in rows:
 if x['checkpoint']=='best' and x['split']=='dev': print(x)
