#!/usr/bin/env python3
"""Redraw the archived seed-5576 projections (no projection refitting)."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('Use a new output directory')
 a.output.mkdir(parents=True)
 for kind in ['pca','tsne']:
  fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
  for row,ds in enumerate(['mosi','mosei']):
   for col,variant in enumerate(['baseline','dynamic']):
    with np.load(R/'analysis/representations/projections'/(ds+'_'+variant+'_projections.npz'),allow_pickle=False) as z:
     xy=z[kind];y=z['labels'];ev=z['explained_variance_ratio']
    ax=axs[row,col]
    for mask,label,color in [(y<0,'Negative','#b34a55'),(y==0,'Neutral','#baa577'),(y>0,'Positive','#448da5')]:
     ax.scatter(xy[mask,0],xy[mask,1],s=5,alpha=.45,c=color,label=label,linewidths=0)
    ax.set_title(ds.upper()+' / '+variant+' / n='+str(len(y)))
    ax.set_xlabel(('PC1 (%.1f%%)'%(ev[0]*100)) if kind=='pca' else 't-SNE 1')
    ax.set_ylabel(('PC2 (%.1f%%)'%(ev[1]*100)) if kind=='pca' else 't-SNE 2')
    ax.grid(alpha=.15)
  axs[0,0].legend(fontsize=8)
  fig.savefig(a.output/(kind+'.png'),dpi=180);plt.close(fig)
 print('Redrawn stored projections:',a.output)
if __name__=='__main__':main()
