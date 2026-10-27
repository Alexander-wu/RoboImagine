import hashlib,json,os
from pathlib import Path
D=Path('/opt/roboimagine-workspace')
ROOT=D/'experiments/roboimagine_rq4_matched30_20261026/v1'
SRC=ROOT/'source'
BASE=D/'experiments/r2_separate_controls_20261012/rt1_hybrid500'
ARMS={'current_only':(1,False),'w1':(1,True),'w3':(3,True),'w6':(6,True),'w12':(12,True)}
HORIZONS=[16,32,48,64,96]
GPUS=['GPU-REPLACE-UUID-1','GPU-REPLACE-UUID-2','GPU-REPLACE-UUID-3','GPU-REPLACE-UUID-4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.partial')
 tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def check():
 m=read(SRC/'source_manifest.json')
 for n,h in m.items():assert sha(SRC/n)==h,('Source hash changed',n)
 p=read(ROOT/'protocol.json');assert p['source_manifest_sha256']==sha(SRC/'source_manifest.json')
 assert sha(ROOT/'inputs/manifest.json')==p['inputs_manifest_sha256']
 assert sha(Path(p['checkpoint'])/'receipt.json')==p['checkpoint_receipt_sha256']
 return p
