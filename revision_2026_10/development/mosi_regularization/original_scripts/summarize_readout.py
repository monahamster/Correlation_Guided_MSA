from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;d=json.loads((R/'word_readout_diagnostic_v2.json').read_text());keys=['text_audio','text_visual','audio_visual']
configs=[('Random initialization','word_sentence','initial','word_average_auc'),('Sentence-only training','sentence','best','word_average_auc'),('Word+sentence: cosine of means','word_sentence','best','sentence_auc'),('Word+sentence: mean word cosines','word_sentence','best','word_average_auc')]
summary=[];fig,ax=plt.subplots(figsize=(9.6,4.7),layout='constrained');colors=['#999999','#dd9d43','#3579a8','#269d78']
for i,((label,family,checkpoint,measure),color) in enumerate(zip(configs,colors)):
 rows=[x for x in d['rows'] if x['family']==family and x['checkpoint']==checkpoint];values=np.array([[x['pairs'][k][measure] for k in keys] for x in rows]);mean=values.mean(0);sd=values.std(0,ddof=1)
 summary.append(dict(label=label,seeds=[x['seed'] for x in rows],mean=dict(zip(keys,mean.tolist())),sample_std=dict(zip(keys,sd.tolist()))))
 ax.errorbar(np.arange(3)+(i-1.5)*.10,mean,yerr=sd,fmt='o',capsize=3,label=label,color=color)
ax.axhline(.5,color='gray',linestyle='--',lw=1);ax.set(xticks=range(3),xticklabels=['Text-Audio','Text-Visual','Audio-Visual'],ylabel='Same-length validation pairing AUC',ylim=(.45,.75),title='MOSI: local training and aligned-word correlation readout\nExisting selected checkpoints; mean +/- sample SD over three training seeds');ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.2)
for ext in ['png','svg']:fig.savefig(R/('word_readout_comparison.'+ext),dpi=145)
(R/'word_readout_comparison.json').write_text(json.dumps(dict(summary=summary,scope='Each seed uses the checkpoint previously selected by sentence validation loss. No selection of last checkpoints from diagnostic results; no test.'),indent=2)+'\n')
print('READOUT_COMPARISON_SAVED')
