#!/usr/bin/env python3
"""Validate the release inventory and compile Python without importing or training."""
from pathlib import Path
import hashlib,json,re,ast
R=Path(__file__).resolve().parents[1]
def main():
 provenance=json.loads((R/'provenance.json').read_text());errors=[]
 for entry in provenance:
  p=R/entry['release_path']
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:errors.append('Source/hash mismatch: '+entry['release_path'])
 py=0
 for f in R.rglob('*.py'):
  if any(x in f.parts for x in ['outputs','work','__pycache__']):continue
  try:ast.parse(f.read_text());py+=1
  except SyntaxError as e:errors.append(str(f.relative_to(R))+': '+str(e))
 for f in R.rglob('*'):
  if not f.is_file() or any(x in f.parts for x in ['outputs','work','__pycache__']):continue
  if f.suffix in ['.pt','.pth','.bin','.ckpt','.pkl','.safetensors','.mp4','.wav','.tex','.bib']:errors.append('Excluded asset found: '+str(f.relative_to(R)))
  if f.stat().st_size>20*1024**2:errors.append('Oversize file: '+str(f.relative_to(R)))
  if f.suffix in ['.py','.json','.jsonl','.sh','.md','.txt']:
   t=f.read_text(errors='replace')
   patterns=[r'gh[pousr]_[A-Za-z0-9]{25,}',r'github_pat_[A-Za-z0-9_]{25,}',r'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',r'https://[^/\s:@]+:[^/\s@]+@']
   if any(re.search(x,t) for x in patterns):errors.append('Potential credential in '+str(f.relative_to(R)))
 registry=json.loads((R/'registry.json').read_text())
 report={'archived_files_verified':len(provenance),'python_sources_parsed':py,'runs':len(registry['runs']),'snapshots':len(registry['snapshots']),'errors':errors}
 print(json.dumps(report,indent=2));raise SystemExit(bool(errors))
if __name__=='__main__':main()
