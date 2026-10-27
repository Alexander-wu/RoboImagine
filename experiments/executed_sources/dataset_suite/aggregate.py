"""All outcomes, paired trajectory comparisons; no selection by score."""
from runner import ROOT,read,write
import numpy as np
plan=read(ROOT/'plan.json');groups=plan['comparisons'];out={'state':'SUCCEEDED','scope':plan['scope'],'groups':{},'limitations':plan['limitations']}
for name,paths in groups.items():
 runs={k:read(p) for k,p in paths.items()};ids=[x['id'] for x in next(iter(runs.values()))['evaluation']]
 assert len(ids)==128
 assert all([r['id'] for r in v['evaluation']]==ids for v in runs.values())
 inputs=[x['prepared_input_sha256'] for x in next(iter(runs.values()))['evaluation']]
 assert all([r['prepared_input_sha256'] for r in v['evaluation']]==inputs for v in runs.values())
 report={'means':{},'cost':{},'comparisons':{}}
 for arm,v in runs.items():
  rs=v['evaluation'];report['means'][arm]={str(h):{k:float(np.mean([r['metrics'][str(h)][k] for r in rs])) for k in rs[0]['metrics'][str(h)]} for h in [10,20,32]}
  report['cost'][arm]={k:v.get(k) for k in ['train_seconds','eval_seconds','peak_train_allocated_GiB','peak_allocated_GiB','updates','global_batches']}
  root=__import__('pathlib').Path(paths[arm]).parent
  logs=sorted(root.glob('rank*/train.jsonl'))
  if logs:
   entries=[__import__('json').loads(s) for p in logs for s in p.read_text().splitlines()]
   report['cost'][arm].update(actual_groups=len(entries),judge_seconds_sum=sum(r.get('judge_seconds',0) for r in entries),metric_seconds_sum=sum(r.get('metric_seconds',0) for r in entries),candidate_seconds_sum=sum(r.get('candidate_seconds',0) for r in entries),cost_note='Summed GPU-worker times differ from wall time; allocator peak excludes judge subprocess. Missing historical timing stays null.')
 for base in ['sft','metric','diverse','full_text']:
  if base not in runs:continue
  for arm in runs:
   if arm==base:continue
   pair={}
   for h in [10,20,32]:
    pair[str(h)]={}
    for k in runs[arm]['evaluation'][0]['metrics'][str(h)]:
     delta=np.array([a['metrics'][str(h)][k]-b['metrics'][str(h)][k] for a,b in zip(runs[arm]['evaluation'],runs[base]['evaluation'])])
     rng=np.random.default_rng(20261013);boot=delta[rng.integers(0,128,(5000,128))].mean(1)
     pair[str(h)][k]={'delta':float(delta.mean()),'paired_episode_ci95':np.quantile(boot,[.025,.975]).tolist()}
   report['comparisons'][arm+'_vs_'+base]=pair
 out['groups'][name]=report
write(ROOT/'summary.json',out)
import csv
with (ROOT/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['experiment','arm','horizon','psnr','ssim','lpips','mae'])
 for group,v in out['groups'].items():
  for arm,hs in v['means'].items():
   for h,m in hs.items():w.writerow([group,arm,h,*[m.get(k) for k in ['psnr','ssim','lpips','mae']]])
print('SUMMARY_COMPLETE',ROOT/'summary.json',flush=True)
