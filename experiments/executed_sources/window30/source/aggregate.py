"""Require the complete paired grid; bootstrap independent trajectories only."""
import csv
import numpy as np
from common_new import *
p=check();ph=sha(ROOT/'protocol.json');rng=np.random.default_rng(20261026)
boot=rng.integers(0,30,size=(10000,30));table=[];per_case=[];generation_count=0
def estimate(v):
 a=np.asarray(v,dtype=float);assert a.shape==(30,) and np.isfinite(a).all()
 lo,hi=np.quantile(a[boot].mean(axis=1),[.025,.975])
 return dict(mean=float(a.mean()),ci95=[float(lo),float(hi)])
for rank in range(4):
 for gate in ['generation_gate','metric_gate']:
  r=read(ROOT/f'workers/rank{rank}/{gate}.json');assert r['state']=='PASS' and r['protocol_sha256']==ph
 for H in HORIZONS:
  for arm in ARMS:assert read(ROOT/f'workers/rank{rank}/warm_h{H}_{arm}.json')['state']=='PASS'
for H in HORIZONS:
 for arm in ARMS:
  values=[]
  for row in p['cases']:
   i=row['index'];qp=ROOT/f'quality/h{H}/{arm}/case_{i:03d}.json';q=read(qp)
   assert q['id']==row['id'] and q['protocol_sha256']==ph and q['input_sha256']==row['sha256']
   for k,v in q['per_frame'].items():assert len(v)==H and abs(float(np.mean(v))-q['metrics'][k])<1e-12
   repeats=[];token_hashes=[]
   for rep in range(3):
    path=ROOT/f'generation/h{H}/{arm}/case_{i:03d}_r{rep}.json';r=read(path)
    assert r['id']==row['id'] and r['protocol_sha256']==ph and r['input_sha256']==row['sha256']
    assert len(r['profile']['frames'])==H
    repeats.append(r['profile']);token_hashes.append(r['tokens_sha256']);generation_count+=1
    if rep==0:
     assert q['generation_sha256']==sha(path) and sha(r['artifact']['path'])==r['artifact']['sha256']
   assert len(set(token_hashes))==1
   v=dict(index=i,id=row['id'],horizon=H,arm=arm,**q['metrics'],
     latency_s=float(np.mean([r['total_wall_seconds'] for r in repeats])),
     peak_GiB=float(np.mean([r['peak_allocated_bytes']/2**30 for r in repeats])),
     reserved_GiB=float(np.mean([r['peak_reserved_bytes']/2**30 for r in repeats])))
   v['seconds_per_frame']=v['latency_s']/H;values.append(v);per_case.append(v)
  keys=list(q['metrics'])+['latency_s','peak_GiB','reserved_GiB','seconds_per_frame']
  table.append(dict(horizon=H,arm=arm,n=30,repeats=3,statistics={k:estimate([v[k] for v in values]) for k in keys}))
assert generation_count==2250 and len(per_case)==750
assert len(list((ROOT/'generation').glob('h*/*/case_*_r*.json')))==2250
assert len(list((ROOT/'quality').glob('h*/*/case_*.json')))==750
deltas=[]
for H in HORIZONS:
 base=[v for v in per_case if v['horizon']==H and v['arm']=='w6']
 for arm in ARMS:
  test=[v for v in per_case if v['horizon']==H and v['arm']==arm]
  assert [v['id'] for v in test]==[v['id'] for v in base]==[r['id'] for r in p['cases']]
  deltas.append(dict(horizon=H,arm=arm,reference='w6',statistics={k:estimate([a[k]-b[k] for a,b in zip(test,base)]) for k in ['psnr','ssim','lpips','mae','latency_s','peak_GiB']}))
out=dict(state='SUCCEEDED',protocol=p,protocol_sha256=ph,generation_records=2250,quality_records=750,
 unique_trajectories=30,table=table,paired_differences=deltas,per_case=per_case,
 bootstrap_resamples=10000,bootstrap_seed=20261026,visual_review_pending=True)
write(ROOT/'summary.json',out)
with (ROOT/'table.csv').open('w',newline='') as f:
 fields=['horizon','arm','n','psnr','ssim','lpips','mae','latency_s','peak_GiB'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in table:w.writerow(dict(horizon=r['horizon'],arm=r['arm'],n=30,**{k:r['statistics'][k]['mean'] for k in fields[3:]}))
print('AGGREGATE_VERIFIED',2250,750,30,flush=True)
