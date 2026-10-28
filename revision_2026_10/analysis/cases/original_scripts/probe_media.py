import json,requests,datetime,concurrent.futures
from pathlib import Path
R=Path('/data01/lyw/GLoMo_case_analysis_20261022')
rows=json.loads((R/'candidates.json').read_text())
def probe(r):
 url='https://www.youtube.com/oembed';out={'candidate_id':r['candidate_id'],'video_id':r['video_id'],'checked_utc':datetime.datetime.utcnow().isoformat(),'watch_url':r['public_video_url']}
 try:
  q=requests.get(url,params={'url':'https://www.youtube.com/watch?v='+r['video_id'],'format':'json'},timeout=10);out['http_status']=q.status_code
  if q.status_code==200:
   d=q.json();out.update(status='metadata_available_playback_unverified',title=d.get('title'),author_name=d.get('author_name'))
  else:out.update(status='metadata_unavailable_not_proof_of_deleted_video',response_excerpt=q.text[:200])
 except Exception as e:out.update(status='network_error_availability_unknown',error=str(e)[:300])
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 out=list(pool.map(probe,rows))
(R/'public_media_probe.json').write_text(json.dumps(out,indent=2))
u='https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/main/README.md'
s={'url':u,'checked_utc':datetime.datetime.utcnow().isoformat()}
try:
 r=requests.get(u,timeout=10);s['http_status']=r.status_code
 if r.ok:(R/'official_sdk_readme.txt').write_text(r.text)
except Exception as e:s['error']=str(e)
(R/'official_source.json').write_text(json.dumps(s,indent=2))
print('PROBE',[(x['candidate_id'],x['status'],x.get('http_status')) for x in out],flush=True)
