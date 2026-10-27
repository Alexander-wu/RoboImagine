"""Immutable long-horizon protocol, shared by the finite inference workers."""
import hashlib,json,os
from pathlib import Path
D=Path('/opt/roboimagine-workspace')
BASE=D/'experiments/r2_separate_controls_20261012/rt1_hybrid500'
SOURCE=Path(__file__).resolve().parent
OUTPUT=D/'experiments/r2_long_memory_20261024/eval_v2'
INPUTS=D/'experiments/r2_long_memory_20261024/eval_v1/inputs'
COHORTS={'historical_dev64':64,'extended_dev128':128,'heldout_test128':128}
ARMS={
 'full_w1':{'recent_states':1,'retain_anchor':True},
 'full_w3':{'recent_states':3,'retain_anchor':True},
 'full_w6':{'recent_states':6,'retain_anchor':True},
 'full_w12':{'recent_states':12,'retain_anchor':True},
 'recent_w6':{'recent_states':6,'retain_anchor':False},
 'current_only':{'recent_states':1,'retain_anchor':False}}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.partial')
 t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def build_protocol(output=OUTPUT):
 source_manifest=read(BASE/'source_manifest.json')
 for name,h in source_manifest.items():assert sha(BASE/name)==h,('Original source changed',name)
 cp=BASE/'run/checkpoint_000500';receipt=read(cp/'receipt.json')
 assert receipt['updates']==500 and receipt['digest']=='f4a63e643ec4525f7abedd5edbfd4a3ad1b764b184f36e202fde9cc2e46b9aa1'
 for name,h in receipt['files'].items():assert sha(cp/name)==h
 cohorts={}
 for name,h in COHORTS.items():
  path=INPUTS/name/'manifest.json';data=read(path)
  assert data['state']=='READY' and data['horizon']==h
  cases=[]
  for i,row in enumerate(data['cases']):
   assert sha(row['path'])==row['sha256']
   cases.append(dict(row,index=i,prepared_index=row['index']))
  assert len(cases)=={'historical_dev64':18,'extended_dev128':5,'heldout_test128':11}[name]
  cohorts[name]=dict(horizon=h,cases=cases,manifest=str(path),manifest_sha256=sha(path),scope=data['data_scope'])
 ids=[r['id'] for c in cohorts.values() for r in c['cases']]
 assert len(ids)==len(set(ids))
 registry=read(D/'data/single_arm_full_20261007/dataset_registry.json')['rt1']['splits']
 train_path=Path(registry['train']['manifest']);assert sha(train_path)==registry['train']['sha256']
 train_rows=[json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
 assert not set(ids)&{r['id'] for r in train_rows},'Long cohort overlaps the complete logical training split'
 for name,split in [('extended_dev128','dev'),('heldout_test128','test')]:
  path=Path(registry[split]['manifest']);assert sha(path)==registry[split]['sha256']
  rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  eligible={r['id'] for r in rows if r['frames']>=130 and r['text'].strip()}
  assert eligible=={r['id'] for r in cohorts[name]['cases']},'Expected all eligible long episodes'
 config=read(D/'bundle/rt1/models/sft/config.json')
 max_context=max(1280+(v['recent_states']+int(v['retain_anchor']))*93+80 for v in ARMS.values())
 assert max_context<=config['max_position_embeddings']
 source_files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*.py') if '__pycache__' not in p.parts}
 p=dict(version=2,experiment='RT1_R2_Hybrid500_long_horizon_memory',checkpoint=str(cp),checkpoint_digest=receipt['digest'],
  checkpoint_receipt_sha256=sha(cp/'receipt.json'),source_manifest_sha256=sha(BASE/'source_manifest.json'),
  original_complete_sha256=sha(BASE/'run/complete.json'),selection_sha256=sha(BASE/'selection.json'),
  source_files=source_files,cohorts=cohorts,arms=ARMS,beam_width=4,precision='FP32; TF32 disabled',
  shared_inputs='Fixed initial scene encoding; two observed frames; unchanged instruction and matched real planned actions; no future RGB input or GT refresh',
  trained_updates=0,gpus=4,max_position_embeddings=config['max_position_embeddings'],max_prefix_plus_frame_tokens=max_context,
  complete_training_manifest_sha256=registry['train']['sha256'],complete_training_split_overlap=0,
  W12_note='Inference window extended beyond native six-state history; no retraining or position scaling',
  horizons=[32,64,128],metrics='Native float-pixel PSNR/SSIM/LPIPS(VGG)/MAE/MSE; evaluate complete horizon and matched prefixes',
  metric_batching='Native Metrics called in fixed32-frame chunks; concatenate per-frame vectors before prefix means; removes batch-shape reduction noise without loosening parity tolerances',
  selection='All length-eligible episodes in predeclared scopes, before inference; dev and test reported separately; original128 has zero H128-eligible episodes',
  seed='Historical cases:20261011+10000+original_index; new cases:20261024+first8hex(id); deterministic beam',
  bootstrap_seed=20261024,bootstrap_resamples=5000)
 dest=output/'protocol.json'
 if dest.exists():assert read(dest)==p,'Changed protocol cannot resume existing results'
 else:write(dest,p)
 return p
def check_protocol(output=OUTPUT):
 p=read(output/'protocol.json')
 assert sha(BASE/'source_manifest.json')==p['source_manifest_sha256']
 assert sha(BASE/'selection.json')==p['selection_sha256']
 assert sha(BASE/'run/complete.json')==p['original_complete_sha256']
 for name,h in p['source_files'].items():assert sha(SOURCE/name)==h,('Worker source changed',name)
 for c in p['cohorts'].values():assert sha(c['manifest'])==c['manifest_sha256']
 return p
if __name__=='__main__':
 import argparse
 q=argparse.ArgumentParser();q.add_argument('--output',type=Path,default=OUTPUT);a=q.parse_args()
 p=build_protocol(a.output);print('PROTOCOL_READY',sum(len(c['cases']) for c in p['cohorts'].values())*len(p['arms']),flush=True)
