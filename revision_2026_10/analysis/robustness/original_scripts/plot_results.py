from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
data=json.loads((R/'comparison.json').read_text())['summary']
out=R/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
lookup={(x['dataset'],x['variant'],x['kind'],x['level']):x['statistics'] for x in data}
colors={'baseline':'#64748b','constant':'#d97706','dynamic':'#2563eb'}
def curve(ds,var,kind,key):
    levels=[.1,.3,.5] if kind=='audio' else [.2,.4,.6]
    stats=[lookup[(ds,var,'clean',0.)]]+[lookup[(ds,var,kind,x)] for x in levels]
    return [0.]+levels,np.asarray([s[key]['mean'] for s in stats]),np.asarray([s[key]['sd'] for s in stats])
for metric,label,fname in [('performance.mae','MAE','performance'),('mae_degradation','MAE change from clean','degradation')]:
    fig,axs=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for row,ds in enumerate(['mosi','mosei']):
        for col,kind in enumerate(['audio','visual']):
            ax=axs[row,col]
            for var in ['baseline','constant','dynamic']:
                x,y,sd=curve(ds,var,kind,metric);ax.errorbar(x,y,yerr=sd,marker='o',lw=1.6,capsize=3,color=colors[var],label=var.capitalize())
            ax.set(title=ds.upper()+' / '+kind,xlabel='Audio noise standard deviation' if kind=='audio' else 'Visual word masking probability',ylabel=label)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/(fname+'.pdf'));fig.savefig(out/(fname+'.png'),dpi=220);plt.close(fig)
for kind in ['audio','visual']:
    fig,axs=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for row,ds in enumerate(['mosi','mosei']):
        for col in range(3):
            ax=axs[row,col]
            for mod,color in [('t','#475569'),('a','#0891b2'),('v','#be185d')]:
                key=['routing.q_'+mod+'.mean_delta_clean','routing.'+mod+'.gate_l1_from_clean','routing.'+mod+'.fixed_feature_reliability_gate_l1'][col]
                x,y,sd=curve(ds,'dynamic',kind,key);ax.errorbar(x,y,yerr=sd,marker='o',capsize=2,color=color,label={'t':'Text','a':'Audio','v':'Vision'}[mod])
            ax.set(title=ds.upper(),xlabel='Noise standard deviation' if kind=='audio' else 'Word masking probability',ylabel=['Change in correlation score q','Gate L1 change from clean','Gate L1 change from reliability bias\n(features held fixed)'][col])
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/('mechanism_'+kind+'.pdf'));fig.savefig(out/('mechanism_'+kind+'.png'),dpi=220);plt.close(fig)
print('FIGURES_COMPLETE',out)
