from pathlib import Path
import sys,json
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler,normalize
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'scripts'))
from pilot import load,save_json

def main():
 torch.set_num_threads(4);data={};z={};fitted={}
 for split in ['train','dev']:data[split],_=load('mosi',split);z[split]={}
 for mod in ['text','audio','visual']:
  tr=np.stack([r[mod].mean(0).numpy() for r in data['train']]);dv=np.stack([r[mod].mean(0).numpy() for r in data['dev']]);scaler=StandardScaler().fit(tr);pca=PCA(n_components=32,whiten=True,svd_solver='randomized',random_state=20261025).fit(scaler.transform(tr))
  z['train'][mod]=pca.transform(scaler.transform(tr));z['dev'][mod]=pca.transform(scaler.transform(dv));fitted[mod]={'pca_variance_fraction':float(pca.explained_variance_ratio_.sum()),'input_dimensions':tr.shape[1]}
 maps={}
 sys.path.insert(0,str(R/'scripts'));from pilot import donors,P
 for split in ['train','dev']:maps[split]=[donors(np.array([r['length'] for r in data[split]]),'same_length',P['dev_negative_seed']+k) for k in range(5)]
 results=[];arrays={}
 for a in ['text','audio','visual']:
  for b in ['text','audio','visual']:
   if a==b:continue
   model=Ridge(alpha=len(data['train'])*.1,fit_intercept=False).fit(z['train'][a],z['train'][b]);row={'source':a,'target':b,'splits':{}}
   for split in ['train','dev']:
    pred=normalize(model.predict(z[split][a]));target=normalize(z[split][b]);pos=(pred*target).sum(-1);neg=np.stack([(pred*target[d]).sum(-1) for d in maps[split]]);aucs=[roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(pos))],np.r_[pos,v]) for v in neg]
    row['splits'][split]={'auc_mean':float(np.mean(aucs)),'auc_each':[float(v) for v in aucs],'positive_mean':float(pos.mean()),'negative_mean':float(neg.mean())};arrays[a+'_to_'+b+'_'+split+'_positive']=pos;arrays[a+'_to_'+b+'_'+split+'_negative']=neg
   results.append(row);print(a,'->',b,'train/dev',*[round(row['splits'][s]['auc_mean'],4) for s in ['train','dev']],flush=True)
 np.savez_compressed(R/'linear_predictability_arrays.npz',**arrays)
 save_json(R/'linear_predictability.json',{'scope':'Diagnostic of pooled feature cross-modal predictability, not the proposed branch or a sentiment model. Train-only standardization, 32 whitened PCA components, fixed ridge penalty n_train*0.1; no hyperparameter search or test evaluation. Cosine compares predicted target PCA features against actual targets.','fitted_on':'MOSI eligible train only','settings_selected_before_run':True,'sample_counts':{k:len(v) for k,v in data.items()},'feature_projection':fitted,'rows':results})
if __name__=='__main__':main()
