from pathlib import Path
import os,json,hashlib,textwrap
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
P=Path(__file__).resolve().parents[1];F=P/'figures'
R=Path('/data01/lyw/GLoMo_final_evaluation_20261022')
C=json.loads((R/'test_comparison.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
variants=['baseline','constant','dynamic']; colors=['#6C7785','#D39032','#176D9C']
seeds=[42,2026,2027,3407,5576]
def save(fig,name):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(F/(name+'.png'),dpi=160,bbox_inches='tight')
 plt.close(fig)
conf={};groups={}
for mode in ['binary_all','binary_nonzero','seven']:
 fig,axs=plt.subplots(2,2 if mode=='seven' else 3,figsize=(8.6,8.2) if mode=='seven' else (11.6,6.8),layout='constrained')
 fig.get_layout_engine().set(h_pad=.16,w_pad=.08,hspace=.13,wspace=.04)
 n=7 if mode=='seven' else 2
 for row,ds in enumerate(['mosi','mosei']):
  for col,v in enumerate(variants):
   matrices=[]; groupstats=[]
   for seed in seeds:
    arr=np.load(R/'evaluations/test'/f'{ds}_{v}_seed{seed}'/'predictions.npz')
    y=arr['labels'];pred=arr['predictions'];keep=np.ones(len(y),bool)
    if mode=='seven':yt=np.rint(np.clip(y,-3,3)).astype(int)+3;pt=np.rint(np.clip(pred,-3,3)).astype(int)+3;metric='acc7'
    elif mode=='binary_nonzero':keep=y!=0;yt=(y>0).astype(int);pt=(pred>0).astype(int);metric='acc2_non_zero'
    else:yt=(y>=0).astype(int);pt=(pred>=0).astype(int);metric='acc2'
    cm=confusion_matrix(yt[keep],pt[keep],labels=list(range(n)));matrices.append(cm)
    assert abs(np.trace(cm)/cm.sum()-C['tables']['main'][ds][v]['per_seed'][str(seed)][metric])<1e-12
    if mode=='binary_all':
     groupstats.append({str(g):{'n':int((np.sign(y)==g).sum()),'mae':float(np.abs(pred-y)[np.sign(y)==g].mean()),'all_binary_accuracy':float((yt==pt)[np.sign(y)==g].mean())} for g in [-1,0,1]})
   counts=np.mean(matrices,axis=0);pct=counts/counts.sum(axis=1,keepdims=True)*100
   conf[f'{ds}_{v}_{mode}']={'per_seed_counts':np.asarray(matrices).tolist(),'mean_counts':counts.tolist(),'row_percent':pct.tolist(),'mean_accuracy':float(np.trace(counts)/counts.sum())}
   if groupstats:groups[f'{ds}_{v}']=groupstats
   if mode=='seven' and v=='constant':continue
   panelcol=(0 if v=='baseline' else 1) if mode=='seven' else col
   ax=axs[row,panelcol];im=ax.imshow(pct,cmap='Blues',vmin=0,vmax=100)
   for a in range(n):
    for b in range(n):
     label=f'{pct[a,b]:.1f}%' if mode=='seven' else f'{counts[a,b]:.1f}\n{pct[a,b]:.1f}%'
     ax.text(b,a,label,ha='center',va='center',fontsize=8 if n==7 else 10,color='white' if pct[a,b]>55 else '#17212B')
   ticks=list(range(-3,4)) if mode=='seven' else (['Negative','Non-negative'] if mode=='binary_all' else ['Negative','Positive'])
   ax.set(xticks=range(n),yticks=range(n),xticklabels=ticks,yticklabels=ticks,xlabel='Predicted class',ylabel='True class')
   ax.set_title(f'{ds.upper()} | {v.title()}\nAccuracy {np.trace(counts)/counts.sum()*100:.2f}%',fontsize=10)
 fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.75,label='Within true class (%)',pad=.02)
 save(fig,'confusion_'+mode)
(P/'confusion_statistics.json').write_text(json.dumps(conf,indent=2))
(P/'label_group_statistics.json').write_text(json.dumps(groups,indent=2))
rob=json.loads(Path('/data01/lyw/GLoMo_robustness_20261022/comparison.json').read_text())['summary']
def rr(ds,v,kind,level):
 return next(x for x in rob if (x['dataset'],x['variant'],x['kind'],x['level'])==(ds,v,kind,float(level)))
for metric,name,ylabel in [('performance.mae','robustness_mae','MAE'),('mae_degradation','robustness_change','MAE change from clean')]:
 fig,axs=plt.subplots(2,2,figsize=(9.8,6.5),layout='constrained')
 for i,ds in enumerate(['mosi','mosei']):
  for k,kind in enumerate(['audio','visual']):
   ax=axs[i,k];levels=[0,.1,.3,.5] if kind=='audio' else [0,.2,.4,.6]
   for v,color in zip(variants,colors):
    rows=[rr(ds,v,'clean' if x==0 else kind,x) for x in levels]
    vals=[r['statistics'].get(metric,{'mean':0,'sd':0}) for r in rows]
    means=np.array([x['mean'] for x in vals]);sds=np.array([x['sd'] for x in vals])
    ax.plot(levels,means,'o-',color=color,label=v.title(),ms=4,lw=1.6)
    ax.fill_between(levels,means-sds,means+sds,color=color,alpha=.1)
   if name=='robustness_change':ax.axhline(0,lw=.7,color='#333333',linestyle='--')
   ax.set(title=f'{ds.upper()} | '+('Audio noise' if kind=='audio' else 'Visual masking'),xlabel=('Noise standard deviation' if kind=='audio' else 'Mask probability'),ylabel=ylabel,xticks=levels)
   ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(alpha=.15)
 axs[0,0].legend(frameon=False,ncol=3,fontsize=8)
 save(fig,name)
fig,axs=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
for col,kind in enumerate(['audio','visual']):
 m='a' if kind=='audio' else 'v';levels=[0,.1,.3,.5] if kind=='audio' else [0,.2,.4,.6]
 for ds,color in [('mosi','#9A4D83'),('mosei','#176D9C')]:
  rows=[rr(ds,'dynamic','clean' if level==0 else kind,level) for level in levels]
  vals=[r['statistics'].get('routing.q_'+m+'.mean_delta_clean',{'mean':0})['mean'] for r in rows]
  axs[0,col].plot(levels,vals,'o-',color=color,label=ds.upper(),lw=1.7)
  # The same frozen branch is shared by downstream seeds; plot means only.
  vals=[r['statistics'].get('routing.'+m+'.fixed_feature_reliability_gate_l1',{'mean':0})['mean'] for r in rows]
  axs[1,col].plot(levels,vals,'o-',color=color,label=ds.upper(),lw=1.7)
 axs[0,col].set(title='Audio noise' if kind=='audio' else 'Visual masking',ylabel='Mean correlation change from clean')
 axs[0,col].axhline(0,color='#666666',lw=.7,linestyle='--')
 axs[1,col].set(ylabel='Gate L1 change from correlation bias',xlabel='Noise standard deviation' if kind=='audio' else 'Mask probability')
 for ax in axs[:,col]:ax.set_xticks(levels);ax.grid(alpha=.15);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
axs[0,0].legend(frameon=False)
save(fig,'correlation_routing_response')
eff=json.loads(Path('/data01/lyw/GLoMo_efficiency_20261022/comparison.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(11.8,3.6),layout='constrained')
for ax,key,title in zip(axs,['parameters','median_ms','peak_allocated_mib'],['Resident parameters (M)','Batch-32 median latency (ms)','Batch-32 peak allocated memory (MiB)']):
 for i,v in enumerate(variants):
  vals=[next(x for x in eff['rows'] if x['dataset']==ds and x['variant']==v and x['batch_size']==32)[key] for ds in ['mosi','mosei']]
  vals=np.array(vals)/(1e6 if key=='parameters' else 1)
  bars=ax.bar(np.arange(2)+(i-1)*.24,vals,width=.22,color=colors[i],label=v.title())
  for b,value in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,value,f'{value:.1f}',ha='center',va='bottom',fontsize=7)
 ax.set(xticks=[0,1],xticklabels=['MOSI','MOSEI'],title=title)
 ax.set_ylim(0,ax.get_ylim()[1]*1.13);ax.grid(axis='y',alpha=.15)
axs[0].legend(frameon=False,fontsize=8)
save(fig,'efficiency')
cases=json.loads(Path('/data01/lyw/GLoMo_case_analysis_20261022/media_recovery_20261023/selected_six.json').read_text())
# Dataset columns, benefit / loss / similar rows. Preserve existing preselected order.
fig=plt.figure(figsize=(11.5,12),layout='constrained');outer=fig.add_gridspec(3,2)
case_stats=[]
for i,c in enumerate(cases):
 ds=c['dataset'];row=sum(x['dataset']==ds for x in cases[:i]);col=0 if ds=='mosi' else 1
 gs=outer[row,col].subgridspec(3,3,height_ratios=[1,0.9,1.25],hspace=.08)
 for k,path in enumerate(c['media_measurements']['frames']):
  ax=fig.add_subplot(gs[0,k]);ax.imshow(plt.imread(path));ax.axis('off')
  if k==1:ax.set_title(ds.upper()+' | '+['Lower error','Higher error','Similar predictions'][row],fontsize=9)
 ax=fig.add_subplot(gs[1,:]);ax.axis('off')
 snippet=' '.join(c['model_input_text'].split())
 if len(snippet)>155:snippet=snippet[:152]+'...'
 ax.text(0,1,'\n'.join(textwrap.wrap(snippet,73)),va='top',fontsize=8)
 ax.text(0,.02,'Label '+f"{c['label']:+.2f}"+' | '+c['sample_id'],va='bottom',fontsize=6.6,color='#45515C')
 ax=fig.add_subplot(gs[2,:])
 means=[c['models'][v]['prediction_mean'] for v in variants];sds=[c['models'][v]['prediction_sd'] for v in variants]
 for k,(mean,color) in enumerate(zip(means,colors)):ax.errorbar(mean,k,xerr=sds[k],fmt='o',color=color,elinewidth=1.5,capsize=3,ms=4)
 ax.axvline(c['label'],color='#202C36',ls='--',lw=1)
 ax.set(yticks=range(3),yticklabels=[v.title() for v in variants],xlabel='Predicted sentiment: mean +/- SD',xlim=(-3,3),ylim=(2.5,-.5))
 ax.grid(axis='x',alpha=.15)
 case_stats.append({'dataset':ds,'id':c['sample_id'],'label':c['label'],'means':means,'sds':sds,'maes':[c['models'][v]['mean_absolute_error'] for v in variants]})
save(fig,'case_examples')
(P/'case_summary.json').write_text(json.dumps(case_stats,indent=2))
print('Confusion, robustness, efficiency and case figures complete',flush=True)
