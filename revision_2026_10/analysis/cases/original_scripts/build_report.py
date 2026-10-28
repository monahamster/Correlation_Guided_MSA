import json,html,hashlib,datetime,textwrap
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data01/lyw/GLoMo_case_analysis_20261022'); rows=json.loads((R/'candidates.json').read_text());(R/'figures').mkdir(exist_ok=True)
T=['t','a','v']; V=['baseline','constant','dynamic']; checks=[]
# Replay stored clean logits only: no model or dataset changes.
for r in rows:
 d=r['models']['dynamic']['seed5576_trace']; r['routing_intervention']={}
 for m in T:
  b=torch.tensor(d[m+'_base_logits'],dtype=torch.float32,device='cuda:0')[None,:];bias=torch.tensor(d[m+'_rel_bias'],dtype=torch.float32,device='cuda:0')[None,:]
  def gate(logits):
   values,indices=logits.topk(3,dim=1);values=values[:,:2];indices=indices[:,:2];return torch.zeros_like(logits).scatter(1,indices,torch.softmax(values,dim=1))
  g=gate(b+bias)[0].cpu().numpy(); stored=np.array(d[m+'_gates']);z=gate(b)[0].cpu().numpy();neutral=gate(b+bias/float(d['s_'+m]))[0].cpu().numpy();err=float(np.max(np.abs(g-stored)));assert err<1e-6
  sort=np.sort(b[0].cpu().numpy());actual_sort=np.sort((b+bias)[0].cpu().numpy());tie=bool(sort[0]==sort[1] or actual_sort[0]==actual_sort[1])
  r['routing_intervention'][m]={'actual_gates':g.tolist(),'zero_reliability_bias_gates':z.tolist(),'neutral_scale_one_gates':neutral.tolist(),'sample_dependent_scale_gate_l1':float(np.abs(g-neutral).sum()),'gate_l1_difference':float(np.abs(g-z).sum()),'actual_experts_zero_based':np.flatnonzero(g).tolist(),'zero_bias_experts_zero_based':np.flatnonzero(z).tolist(),'top2_boundary_tie':tie,'actual_gate_replay_max_error':err,'scope':'Fixed clean gate features; removes correlation-dependent MoE bias only. No final prediction intervention, no fusion intervention.'}
  checks.append(err)
(R/'candidates_with_routing.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
(R/'routing_validation.json').write_text(json.dumps({'device':torch.cuda.get_device_name(0),'candidate_count':len(rows),'modality_replays':len(checks),'max_actual_gate_replay_error':max(checks),'boundary_ties':sum(x['top2_boundary_tie'] for r in rows for x in r['routing_intervention'].values()),'prediction_intervention_performed':False},indent=2))
probes={x['candidate_id']:x for x in json.loads((R/'public_media_probe.json').read_text())}
styles='<style>body{font:16px/1.55 sans-serif;max-width:1250px;margin:30px auto;color:#182433;background:#f7f8fc}article{background:white;padding:24px;margin:22px 0;border:1px solid #d9dfeb;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}code{font-size:12px}blockquote{border-left:4px solid #4676bb;padding:12px;background:#f3f6fa}textarea{width:98%;height:65px}small{color:#566273}</style>'
full=['<!doctype html><meta charset="utf-8"><title>Case candidates</title>'+styles+'<h1>Case candidates: predictions and routing</h1><p>Post-hoc descriptive candidates; 5 training seeds, sample SD. Four pools per dataset, four independent videos per pool. Final six multimodal cases await media review. q is a cross-modal correlation score. Expert indices are local to seed5576.</p>']
blind=['<!doctype html><meta charset="utf-8"><title>Media annotation worksheet</title>'+styles+'<h1>Media observation worksheet</h1><p>Labels, predictions and candidate groups omitted. Record visible and audible events with timestamps before opening the result report. Independent annotation has not yet been performed. Fields are a printable worksheet; browser edits are not saved automatically.</p>']
annotations=[]
for n,r in enumerate(sorted(rows,key=lambda x:(x['dataset'],x['sample_id']))):
 ident='CASE-'+str(n+1).zfill(2);r['annotation_id']=ident
 ann={'annotation_id':ident,'dataset':r['dataset'],'sample_id':r['sample_id'],'video_id':r['video_id'],'start_seconds':r['start_seconds'],'end_seconds':r['end_seconds'],'transcript':r['transcript'],'media_status':'not_acquired','text_cues':None,'audible_cues':None,'visible_cues':None,'cross_modal_relationship':None,'timestamp_verified_against_media':False,'annotator':None};annotations.append(ann)
 link=html.escape(r['public_video_url']); stamp=f"{r['start_seconds']:.3f}-{r['end_seconds']:.3f} s";title=html.escape(r['sample_id']); quote=html.escape(r['transcript'])
 base=f'<h2>{ident} | {r["dataset"].upper()} | {title}</h2><p><a href="{link}">Original public video reference</a> | {stamp}</p><blockquote>{quote}</blockquote>'
 blind.append('<article>'+base+'<p>Media / alignment verified: ____________________</p>'+''.join('<p>'+s+'</p><textarea></textarea>' for s in ['Text evidence','Audible evidence + time','Visible evidence + time','Agreement / difference / unclear + supporting observation'])+'</article>')
 part='<article>'+base+f'<p>Pool: {r["candidate_group"]} | Label: {r["label"]:.3f} | Media probe: {probes[r["candidate_id"]]["status"]}</p><table><tr><th>Model</th><th>Prediction mean +/- SD</th><th>Mean absolute error</th><th>5 seed predictions</th></tr>'
 for v in V:
  z=r['models'][v]; part+=f'<tr><td>{v}</td><td>{z["prediction_mean"]:.4f} +/- {z["prediction_sd"]:.4f}</td><td>{z["mean_absolute_error"]:.4f}</td><td>'+', '.join(f'{p:.3f}' for p in z['predictions_by_seed'])+'</td></tr>'
 part+='</table><p>Seeds: 5576, 42, 2026, 2027, 3407. Error is averaged per seed; it is not the error of the mean prediction.</p><table><tr><th>Modality</th><th>q</th><th>s</th><th>Actual gates [E0,E1,E2]</th><th>Zero-bias gates</th><th>Gate L1 change</th></tr>'
 d=r['models']['dynamic']['seed5576_trace']
 for m in T:
  z=r['routing_intervention'][m];fmt=lambda a:', '.join(f'{x:.5f}' for x in a)
  part+=f'<tr><td>{m.upper()}</td><td>{d["q_"+m]:.5f}</td><td>{d["s_"+m]:.5f}</td><td>{fmt(z["actual_gates"])}</td><td>{fmt(z["zero_reliability_bias_gates"])}</td><td>{z["gate_l1_difference"]:.7f}</td></tr>'
 part+='</table><p>Gate intervention keeps clean features fixed and removes the MoE correlation bias. It does not establish a final-prediction effect. Raw pairwise cosines: '+', '.join(k+'='+f'{d["raw_cosine_"+k]:.5f}' for k in ['ta','tv','av'])+'</p></article>';full.append(part)
(R/'candidate_report.html').write_text('\n'.join(full));(R/'media_annotation_worksheet.html').write_text('\n'.join(blind))
(R/'annotation_template.json').write_text(json.dumps(annotations,indent=2,ensure_ascii=False));(R/'candidates_with_routing.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
for ds in ['mosi','mosei']:
 chosen=[next(r for r in rows if r['candidate_id']==ds+'_'+group+'_1') for group in ['consistent_gain','consistent_loss','agreement']]
 fig,axes=plt.subplots(3,3,figsize=(15,14));fig.subplots_adjust(left=.07,right=.98,bottom=.16,top=.87,hspace=1.1,wspace=.4)
 fig.suptitle(ds.upper()+' | provisional numerical cases',fontsize=20,y=.98)
 fig.text(.07,.935,'Text and model evidence only; media interpretation pending. Prediction bars: mean +/- SD across five seeds.\nRouting: Dynamic seed5576. Solid = actual; outlined = same logits with correlation bias removed.',fontsize=11)
 for row,r in enumerate(chosen):
  ax=axes[row,0];means=[r['models'][v]['prediction_mean'] for v in V];sd=[r['models'][v]['prediction_sd'] for v in V];ax.bar(np.arange(3),means,yerr=sd,color=['#8c98ad','#f0b45a','#3678be'],capsize=4);ax.axhline(r['label'],color='#a52d38',ls='--',label='Label '+f'{r["label"]:.2f}');ax.set_xticks(range(3));ax.set_xticklabels(['Baseline','Constant','Dynamic']);ax.set_ylim(-3.1,3.1);ax.legend(fontsize=9);ax.set_ylabel('Sentiment score');ax.set_title(r['candidate_group'].replace('_',' ')+' | '+r['video_id'],fontsize=11)
  ax.text(0,-.28,'\n'.join(textwrap.wrap(r['transcript'],55)),transform=ax.transAxes,fontsize=9,va='top')
  d=r['models']['dynamic']['seed5576_trace'];ax=axes[row,1];q=[d['q_'+m] for m in T];ax.bar(T,q,color='#4c9f91');ax.set_ylim(0,1);ax.set_title('Cross-modal q; annotation gives scale s',fontsize=11)
  for j,m in enumerate(T):ax.text(j,q[j]+.025,f'q={q[j]:.3f}\ns={d["s_"+m]:.3f}',ha='center',fontsize=9)
  ax=axes[row,2]
  for j,m in enumerate(T):
   z=r['routing_intervention'][m];x=np.arange(3)+j*4;ax.bar(x-.15,z['actual_gates'],width=.3,color=['#3678be','#4c9f91','#bb79a4'][j]);ax.bar(x+.15,z['zero_reliability_bias_gates'],width=.3,facecolor='none',edgecolor='#333333');
  ax.set_xticks([1,5,9]);ax.set_xticklabels(['Text (E0/1/2)','Audio (E0/1/2)','Vision (E0/1/2)'],fontsize=9);ax.set_ylim(0,1.1);ax.set_title('Actual versus zero-bias expert gates',fontsize=11)
 fig.savefig(R/'figures'/('numerical_cases_'+ds+'.png'),dpi=160);fig.savefig(R/'figures'/('numerical_cases_'+ds+'.pdf'));plt.close(fig)
print('REPORT COMPLETE; max routing replay error',max(checks),flush=True)
