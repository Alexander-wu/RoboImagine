"""Predeclared paired cohort; read real frames, never select on outcomes."""
import fcntl,importlib.util,time
import numpy as np
from common_new import *
lock=(ROOT/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
c=read(SRC/'selection_config.json'); spec=c['test_manifest']; assert sha(spec['path'])==spec['sha256']
rows=[json.loads(l) for l in Path(spec['path']).read_text().splitlines() if l.strip()];lookup={r['id']:r for r in rows}
eligible=sorted(r['id'] for r in rows if int(r['frames'])>=98 and r['text'].strip())
assert eligible==c['eligible_ids'] and len(eligible)==38
import random
assert sorted(random.Random(c['seed']).sample(eligible,30))==c['selected_ids']
reg=read(D/'data/single_arm_full_20261007/dataset_registry.json')['rt1']['splits']
for split in ['train','dev']:
 p=Path(reg[split]['manifest']);assert sha(p)==reg[split]['sha256']
 assert not set(c['selected_ids'])&{json.loads(l)['id'] for l in p.read_text().splitlines() if l.strip()}
reader_path=D/'experiments/rt1_mjev_20261011/sft_mjev500_8gpu_v1/code/src/futureworlds/data_readers/rt1.py'
sp=importlib.util.spec_from_file_location('rt1_reader',reader_path);reader=importlib.util.module_from_spec(sp);sp.loader.exec_module(reader)
cases=[]
for i,identity in enumerate(c['selected_ids']):
 r=lookup[identity];assert r['split']=='test'
 dest=ROOT/f'inputs/case_{i:03d}.npz'
 if not dest.exists():
  im,ac,tx=reader.read_images(r,start=0,length=98)
  assert im.shape==(98,256,320,3) and im.dtype==np.uint8
  assert ac.shape==(98,13) and ac.dtype==np.float32 and np.isfinite(ac).all() and tx==r['text']
  dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial')
  with tmp.open('wb') as f:np.savez_compressed(f,image=im,action=ac,instruction=np.array(tx))
  tmp.replace(dest)
 with np.load(dest,allow_pickle=False) as z:
  assert z['image'].shape==(98,256,320,3) and z['image'].dtype==np.uint8
  assert z['action'].shape==(98,13) and np.isfinite(z['action']).all() and str(z['instruction'])==r['text']
 cases.append(dict(index=i,id=identity,path=str(dest),sha256=sha(dest),text=r['text'],original=r,seed=(20261026+int(identity[:8],16))%(2**32)))
 write(ROOT/'prepare_status.json',dict(state='PREPARING',completed=len(cases),total=30,time=time.time()))
 print('INPUT_READY',i,identity,flush=True)
manifest=dict(state='READY',cases=cases,selection=c,reader=str(reader_path),reader_sha256=sha(reader_path),train_dev_overlap=0)
write(ROOT/'inputs/manifest.json',manifest)
cp=BASE/'run/checkpoint_000500';receipt=read(cp/'receipt.json');assert receipt['digest']==c['checkpoint_digest'] and receipt['updates']==500
for n,h in receipt['files'].items():assert sha(cp/n)==h
protocol=dict(version=1,checkpoint=str(cp),checkpoint_digest=receipt['digest'],checkpoint_receipt_sha256=sha(cp/'receipt.json'),
 source_manifest_sha256=sha(SRC/'source_manifest.json'),inputs_manifest_sha256=sha(ROOT/'inputs/manifest.json'),
 cases=cases,horizons=HORIZONS,arms=ARMS,repeats=3,beam_width=4,precision='FP32; TF32 disabled',seed=c['seed'],
 selection=c['selection'],eligible=38,selected=30,split='test',initial_observations=2,trained_updates=0,
 expected_generation_records=2250,expected_quality_records=750,
 timing='Three independent whole-rollout measurements per case, horizon and arm; average within trajectory first. Exclude text cache, model loading, scoring, file IO. Warm each arm/horizon in same process. Exclusive GPU use; no training overlap.',
 memory='Peak PyTorch allocated memory including model/codec/text residency; reserved separate; not full device memory.',
 shared_conditions='Same two initial real frames, real action prefixes and instruction; standalone generation at each horizon; no padding or GT refresh; windows exclude fixed dynamic anchor.',
 confidence='Paired 10000 trajectory bootstrap; timing repeats are not independent trajectories; one trained checkpoint.',
 scope='Thirty randomly selected long-eligible test trajectories shared by all five horizons. Findings apply to this long-eligible cohort, not the entire RT1 distribution.')
if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==protocol
else:write(ROOT/'protocol.json',protocol)
write(ROOT/'prepare_status.json',dict(state='READY',completed=30,total=30,time=time.time()))
print('PREPARATION_READY',flush=True)
