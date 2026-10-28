from pathlib import Path
import os,time,json,statistics,hashlib
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
threadpool_limits(4)
P=Path(__file__).resolve().parents[1];F=P/'figures';seeds=[42,2026,2027,3407,5576]
deadline=time.time()+3600
while not (P/'extraction_complete.json').exists():
 if time.time()>deadline:raise RuntimeError('Feature extraction not complete')
 time.sleep(15)
rows=[];projections={};replay=[]
for ds in ['mosi','mosei']:
 for v in ['baseline','dynamic']:
  for seed in seeds:
   ident=f'{ds}_{v}_seed{seed}'
   path=P/'features/test'/ident/'predictions.npz';d=np.load(path)
   x=d['features'].astype(np.float64);y=d['labels'];nz=y!=0
   assert x.ndim==2 and len(x)==len(y) and np.isfinite(x).all()
   metrics=json.loads((path.parent/'metrics.json').read_text())
   replay.append({'id':ident,'max_prediction_error':metrics['dev_replay_max_abs_diff'],'n':len(y),'dimension':x.shape[1],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
   scores={'dataset':ds,'variant':v,'seed':seed,'silhouette_3class':float(silhouette_score(x,np.sign(y),metric='euclidean')),'silhouette_2class_nonzero':float(silhouette_score(x[nz],(y[nz]>0).astype(int),metric='euclidean'))}
   rows.append(scores);print(ident,scores,flush=True)
   if seed==5576:
    pc=PCA(n_components=min(50,x.shape[1]),svd_solver='full');z=pc.fit_transform(x)
    ts=TSNE(n_components=2,perplexity=30,learning_rate=200,n_iter=1000,init='pca',random_state=5576,method='barnes_hut',n_jobs=4).fit_transform(z)
    np.savez_compressed(P/'features'/f'{ds}_{v}_projections.npz',pca=z[:,:2],tsne=ts,labels=y,sample_ids=d['sample_ids'],explained_variance_ratio=pc.explained_variance_ratio_)
    projections[f'{ds}_{v}']={'pca_variance_2d':pc.explained_variance_ratio_[:2].tolist(),'pca_variance_50d':float(pc.explained_variance_ratio_.sum()),'n':len(y)}
summary={}
for ds in ['mosi','mosei']:
 for v in ['baseline','dynamic']:
  summary[f'{ds}_{v}']={m:{'mean':statistics.mean(r[m] for r in rows if r['dataset']==ds and r['variant']==v),'sample_sd':statistics.stdev(r[m] for r in rows if r['dataset']==ds and r['variant']==v)} for m in ['silhouette_3class','silhouette_2class_nonzero']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
for method in ['pca','tsne']:
 fig,axs=plt.subplots(2,2,figsize=(10,7.6),layout='constrained')
 for row,ds in enumerate(['mosi','mosei']):
  for col,v in enumerate(['baseline','dynamic']):
   d=np.load(P/'features'/f'{ds}_{v}_projections.npz');z=d[method];label=np.sign(d['labels'])
   ax=axs[row,col]
   # Fixed label-independent plotting order to limit systematic occlusion.
   order=np.random.default_rng(5576).permutation(len(z))
   palette={-1:'#B64C5B',0:'#B6AA8B',1:'#25899C'}
   ax.scatter(z[order,0],z[order,1],c=[palette[int(k)] for k in label[order]],s=9,alpha=.55,linewidths=0,rasterized=True)
   if method=='pca':
    ev=d['explained_variance_ratio'];ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)');ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)')
   else:ax.set_xlabel('t-SNE 1');ax.set_ylabel('t-SNE 2')
   ax.set_title(ds.upper()+' | '+v.title()+f' | n={len(z)}',fontsize=10)
   ax.grid(alpha=.12)
 from matplotlib.lines import Line2D
 handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=color,markeredgecolor='none',label=label,markersize=6) for label,color in [('Negative (y<0)','#B64C5B'),('Neutral (y=0)','#B6AA8B'),('Positive (y>0)','#25899C')]]
 fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False)
 fig.savefig(F/('representation_'+method+'.pdf'),bbox_inches='tight');fig.savefig(F/('representation_'+method+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
(P/'representation_statistics.json').write_text(json.dumps({'per_seed':rows,'summary':summary,'projections':projections,'replay':replay,'interpretation':'Silhouettes use raw full-dimensional representations; projection coordinates fitted independently per trained model; colors are ground-truth sign classes.'},indent=2))
print('REPRESENTATION_COMPLETE',flush=True)
