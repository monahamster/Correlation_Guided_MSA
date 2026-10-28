import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
def sentiment_metrics(predictions,labels):
 p=np.asarray(predictions,dtype=np.float64).reshape(-1);y=np.asarray(labels,dtype=np.float64).reshape(-1)
 if len(p)!=len(y) or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Require equally sized nonempty finite arrays')
 nz=y!=0;yt=y>=0;yp=p>=0
 out=dict(n=len(y),n_nonzero=int(nz.sum()),n_zero=int((~nz).sum()),mae=float(np.abs(p-y).mean()),mse=float(np.square(p-y).mean()),corr=float(np.corrcoef(p,y)[0,1]) if p.std()>1e-12 and y.std()>1e-12 else None,acc7=float(np.mean(np.round(np.clip(p,-3,3))==np.round(np.clip(y,-3,3)))),acc2=float(np.mean(yp==yt)),f1=float(f1_score(yt,yp,average='weighted',zero_division=0)),acc2_non_zero=float(np.mean((p[nz]>0)==(y[nz]>0))) if nz.any() else None,f1_non_zero=float(f1_score(y[nz]>0,p[nz]>0,average='weighted',zero_division=0)) if nz.any() else None,confusion2=confusion_matrix(yt,yp,labels=[False,True]).tolist(),confusion2_nonzero=confusion_matrix(y[nz]>0,p[nz]>0,labels=[False,True]).tolist() if nz.any() else None)
 return out
if __name__=='__main__':
 x=sentiment_metrics([-4,0,0,-1,4],[-3,-1,0,1,3])
 assert x['mae']==1 and x['mse']==1.4 and x['acc7']==.6 and x['acc2']==.6 and x['acc2_non_zero']==.75
 assert abs(x['f1']-.6)<1e-12 and x['confusion2']==[[1,1],[1,2]]
 assert sentiment_metrics([0],[0])['acc2']==1
 assert sentiment_metrics([0],[0])['acc2_non_zero'] is None
 assert sentiment_metrics([0,0],[-1,1])['corr'] is None
 for p,y in [([],[]),([float('nan')],[0]),([1,2],[1])]:
  try:sentiment_metrics(p,y)
  except ValueError:pass
  else:raise AssertionError('invalid input accepted')
 print('METRIC_EDGE_CASES_PASS')
