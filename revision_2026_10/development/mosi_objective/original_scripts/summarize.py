from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
refs=[('Mean + position',Path('/data01/lyw/GLoMo_length_pretrain_20261020')),('Mean, no position',Path('/data01/lyw/GLoMo_position_ablation_20261020'))]
trials=[('Max + position',R/'max_pool_with_position'),('Max, no position',R/'max_pool_no_position'),('Word + sentence, no position',R/'word_sentence_triplet')]
pairs=['text_audio','text_visual','audio_visual'];rows=[]
for label,d in refs+trials:
 s=json.loads((d/'runs/mosi_same_length/summary.json').read_text());v=s['dev']['same_length']
 rows.append(dict(configuration=label,root=str(d),best_epoch=s['best_epoch'],dev_loss=v['loss'],auc={k:v['pairs'][k]['auc_mean'] for k in pairs}))
(R/'comparison.json').write_text(json.dumps(dict(scope='MOSI seed5576, 20 epochs; common sentence-level same-length validation selection, five fixed donor maps; no test. Max pooling changes aggregation; word+sentence changes only training objective.',rows=rows),indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
colors=['#8897aa','#2878b5','#7f7f7f','#eda543','#269b78']
for row,col in zip(rows,colors):axs[0].plot(range(3),[row['auc'][k] for k in pairs],'o-',color=col,label=row['configuration'])
axs[0].axhline(.5,color='gray',linestyle='--',lw=1);axs[0].set(xticks=range(3),xticklabels=['Text-Audio','Text-Visual','Audio-Visual'],ylabel='Validation pairing AUC',ylim=(.45,.65),title='Selected checkpoints');axs[0].legend(fontsize=8)
for (label,d),col in zip(refs+trials,colors):
 h=json.loads((d/'runs/mosi_same_length/history.json').read_text());axs[1].plot([x['epoch'] for x in h],[x['dev']['same_length']['loss'] for x in h],color=col,label=label)
axs[1].set(title='Common sentence-level validation loss',xlabel='Epoch',ylabel='Triplet loss',xticks=[1,5,10,15,20])
for ax in axs:ax.grid(axis='y',alpha=.2)
fig.suptitle('MOSI pretraining optimization | one seed, fixed evaluation maps',fontsize=12)
for ext in ['png','svg']:fig.savefig(R/('optimization_comparison.'+ext),dpi=145)
print(json.dumps(rows,indent=2))
