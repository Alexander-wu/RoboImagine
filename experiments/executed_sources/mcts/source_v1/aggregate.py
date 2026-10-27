"""Compare all matched128 RT1 trajectories, with no outcome-based selection."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
D=Path('/opt/roboimagine-workspace/experiments')
sources={
 'SFT':D/'rt1_mjev_20261011/sft_reference_v1/complete.json',
 'Sampling':D/'r2_rt1_ablation_20261015/search_sample/run/complete.json',
 'Ordinary beam':D/'r2_rt1_recovery_20261016/ordinary250_to500_v1/complete.json',
 'Grouped diverse beam':D/'r2_separate_controls_20261012/rt1_hybrid500/run/complete.json',
 'Likelihood-guided MCTS (16)':a.run/'complete.json'}
data={k:read(p) for k,p in sources.items()};baseline=data['SFT'];table=[];arrays={}
for name,run in data.items():
 rows=run['evaluation'];assert len(rows)==128
 assert [r['id'] for r in rows]==[r['id'] for r in baseline['evaluation']]
 assert [r['prepared_input_sha256'] for r in rows]==[r['prepared_input_sha256'] for r in baseline['evaluation']]
 assert run['initial_digest']==baseline['initial_digest']
 arrays[name]={};mean={}
 for metric in ['psnr','ssim','lpips','mae','mse']:
  vals=np.asarray([r['per_frame'][metric] for r in rows]);assert vals.shape==(128,32) and np.isfinite(vals).all()
  arr=vals.mean(1);arrays[name][metric]=arr;mean[metric]=float(arr.mean())
  assert np.allclose(arr,[r['metrics']['32'][metric] for r in rows],rtol=0,atol=1e-7)
 table.append(dict(method=name,**mean))
rng=np.random.default_rng(20261025);idx=rng.integers(0,128,(5000,128));paired={};new=arrays['Likelihood-guided MCTS (16)']
for name,values in arrays.items():
 if name=='Likelihood-guided MCTS (16)':continue
 paired[name]={}
 for key,arr in values.items():
  delta=new[key]-arr;ci=np.quantile(delta[idx].mean(1),[.025,.975])
  paired[name][key]=dict(mcts_minus_comparator=float(delta.mean()),paired_trajectory_bootstrap95=ci.tolist())
run=data['Likelihood-guided MCTS (16)'];traces=[]
for f in sorted(a.run.glob('rank*/train.jsonl')):
 traces.extend(json.loads(line) for line in f.read_text().splitlines())
assert len(traces)==run['global_batches']*8 and len({(r['batch'],r['virtual_rank']) for r in traces})==len(traces)
cost=dict(train_seconds=run['train_seconds'],train_gpu_hours=run['train_gpu_hours'],gpu_count=4,
 complete_groups=len(traces),simulations_per_group=16,returned_candidates=4,
 total_model_forward_calls=sum(r['trace']['counters']['model_forward_calls'] for r in traces),
 total_actor_token_logit_rows=sum(r['trace']['counters']['actor_token_logit_rows'] for r in traces),
 candidate_seconds_sum_over_groups=sum(r['candidate_seconds'] for r in traces),
 judge_seconds_sum_over_groups=sum(r['judge_seconds'] for r in traces),
 mean_max_tree_depth=float(np.mean([r['trace']['max_tree_depth'] for r in traces])),
 peak_train_allocated_GiB=run['peak_train_allocated_GiB'],
 note='Same500updates/globalbatch8; different MCTS search budget and physical GPU parallelism. Group component times overlap across4 ranks; do not add them to wall time.')
a.output.mkdir(parents=True,exist_ok=False)
out=dict(state='SUCCEEDED',table=table,paired_mcts_minus_other=paired,cost=cost,source_files={k:dict(path=str(p),sha256=sha(p)) for k,p in sources.items()},protocol=run['protocol'],limitations=['One training seed; bootstrap reflects trajectory uncertainty only.','Likelihood-guided frame-prefix UCT16 is one specified MCTS instantiation, not all MCTS methods.','Do not call this equal-compute; nominal16 rollouts differ from8-path beam.','No dev/test tuning or favorable-sample selection.'])
(a.output/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
with (a.output/'table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print(json.dumps({'state':'SUCCEEDED','table':table,'cost':cost},indent=2))
