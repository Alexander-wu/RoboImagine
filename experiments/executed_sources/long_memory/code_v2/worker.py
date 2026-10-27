"""One explicitly assigned GPU; frozen weights, fail-closed resumable cases."""
import argparse,fcntl,os,sys,time,traceback
from pathlib import Path
from common_long import BASE,OUTPUT,read,write,sha,check_protocol
from metric_chunks import score_in_fixed_chunks
from decoder.window_decoder import predict_with_window,validate_trace
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--rank',type=int,required=True);p.add_argument('--shards',type=int,default=4);p.add_argument('--parity-only',action='store_true');a=p.parse_args()
assert a.shards==4 and 0<=a.rank<4
R=a.output/f'workers/rank{a.rank}';R.mkdir(parents=True,exist_ok=True)
lock=(R/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
protocol=check_protocol(a.output);protocol_sha=sha(a.output/'protocol.json')
sys.path.insert(0,str(BASE))
from common import load_initial,setup,pixels,digest
import numpy as np
import torch
from safetensors.torch import load_file
from futureworlds.evaluation import Metrics
def status(**x):write(R/'status.json',dict(rank=a.rank,updated_at=time.time(),**x))
def seed(row):
 return 20261011+10000+row['original_index'] if row['original_index'] is not None else (20261024+int(row['id'][:8],16))%(2**32)
def load_input(row,H):
 assert sha(row['path'])==row['sha256']
 with np.load(row['path'],allow_pickle=False) as z:im=z['image'].copy();ac=z['action'].copy();text=str(z['instruction'])
 assert im.shape==(H+2,256,320,3) and im.dtype==np.uint8
 assert ac.shape==(H+2,13) and np.isfinite(ac).all() and text==row['text']
 return im,ac,text
def valid_metrics(s,H):
 assert {'psnr','ssim','lpips','mae','mse'}==set(s['per_frame'])
 for k,v in s['per_frame'].items():
  assert len(v)==H and np.isfinite(v).all()
  for h in [32,64,128]:
   if h<=H:assert abs(float(np.mean(v[:h]))-s['metrics'][str(h)][k])<1e-6
def paths(cohort,arm,row):
 d=a.output/'results'/cohort/arm;d.mkdir(parents=True,exist_ok=True)
 return d/f"case_{row['index']:03d}.json",d/f"case_{row['index']:03d}.npz"
def validate_record(rec,dest,cohort,arm,row,H):
 assert (rec['id'],rec['index'],rec['cohort'],rec['arm'],rec['horizon'])==(row['id'],row['index'],cohort,arm,H)
 assert rec['protocol_sha256']==protocol_sha and rec['input_sha256']==row['sha256']
 assert rec['checkpoint_digest']==protocol['checkpoint_digest'] and sha(dest)==rec['artifact_sha256']
 validate_trace(rec['trace'],horizon=H,**protocol['arms'][arm]);valid_metrics(rec,H)
def run_case(cohort,arm,row,check_native_prefix=False):
 H=protocol['cohorts'][cohort]['horizon'];rp,dest=paths(cohort,arm,row)
 if rp.exists():
  rec=read(rp);validate_record(rec,dest,cohort,arm,row,H)
  if check_native_prefix:assert rec.get('native_H32_prefix_parity',{}).get('state')=='PASS'
  return rec
 im,ac,text=load_input(row,H)
 pipe.text_cache.clear();pipe.condition(text)  # warm text cache outside timed inference
 setup(seed(row));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
 result=predict_with_window(pipe,im[:2].copy(),ac[:H+1].copy(),text,horizon=H,beam_width=4,**protocol['arms'][arm])
 assert result['tokens'].shape==(1,H,80)
 torch.cuda.synchronize();seconds=time.monotonic()-tick;peak=torch.cuda.max_memory_allocated()/2**30
 pred=np.asarray(result['prediction'],dtype='float32');assert pred.shape==(H,256,320,3) and np.isfinite(pred).all()
 validate_trace(result['trace'],horizon=H,**protocol['arms'][arm])
 scored=score_in_fixed_chunks(evaluator,pred,im[2:],[h for h in [32,64,128] if h<=H]);valid_metrics(scored,H)
 parity=None
 if check_native_prefix:
  assert arm=='full_w6'
  setup(seed(row));native=pipe.predict(im[:2].copy(),ac[:33].copy(),text,horizon=32,beam_width=4,memory='full')
  assert np.array_equal(result['tokens'][:,:32],native['tokens']),'Long horizon changed the native first32 tokens'
  error=float(np.max(np.abs(pred[:32]-native['prediction'])));assert error<=1e-6,('Prefix pixel mismatch',error)
  ns=evaluator(native['prediction'],im[2:34],[32])
  metric_errors={k:float(np.max(np.abs(np.asarray(scored['per_frame'][k][:32])-ns['per_frame'][k]))) for k in ns['per_frame']}
  write(R/'long_prefix_diagnostic.json',dict(id=row['id'],max_pixel_abs_error=error,metric_max_abs_errors=metric_errors))
  assert all(v<=1e-6 for v in metric_errors.values()),metric_errors
  parity=dict(state='PASS',exact_tokens=True,max_pixel_abs_error=error,metric_atol=1e-6,metric_max_abs_errors=metric_errors)
  del native,ns
 tmp=dest.with_name(dest.name+'.partial')
 with tmp.open('wb') as f:np.savez_compressed(f,prediction=pred,tokens=result['tokens'],ground_truth=im[2:],observations=im[:2],instruction=np.array(text))
 tmp.replace(dest)
 rec=dict(id=row['id'],index=row['index'],cohort=cohort,arm=arm,horizon=H,protocol_sha256=protocol_sha,input_sha256=row['sha256'],
  checkpoint_digest=protocol['checkpoint_digest'],artifact_sha256=sha(dest),metrics=scored['metrics'],per_frame=scored['per_frame'],trace=result['trace'],
  inference_seconds=seconds,peak_allocated_GiB=peak,scoring_and_validation_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
  native_H32_prefix_parity=parity,updated_at=time.time(),wall_seconds=time.monotonic()-tick)
 write(rp,rec);print('LONG_MEMORY_CASE',cohort,arm,row['index'],H,round(seconds,2),flush=True)
 pipe.text_cache.clear();return rec
try:
 torch.cuda.set_device(0);setup(20261024);status(state='LOADING_FROZEN_CHECKPOINT',cases=0)
 pipe=load_initial();cp=Path(protocol['checkpoint']);receipt=read(cp/'receipt.json')
 assert pipe.action_dim==13
 assert sha(cp/'receipt.json')==protocol['checkpoint_receipt_sha256']
 for filename,target in [('model.safetensors',pipe.model.core),('text_adapters.safetensors',pipe.model.adapters)]:
  assert sha(cp/filename)==receipt['files'][filename]
  target.load_state_dict(load_file(str(cp/filename),device='cuda:0'),strict=True)
 actor=pipe.model.eval().requires_grad_(False)
 assert digest(actor)==protocol['checkpoint_digest'] and all(v.dtype==torch.float32 for v in actor.parameters())
 assert actor.core.config.max_position_embeddings>=protocol['max_prefix_plus_frame_tokens']
 evaluator=Metrics('cuda:0')
 if a.parity_only:
  status(state='VERIFYING_ARCHIVED_H32',cases=0)
  index=a.rank;row=read(BASE/'selection.json')['dev'][index];im,ac,text=pixels(row)
  setup(20261011+10000+index)
  result=predict_with_window(pipe,im[:2].copy(),ac[:33].copy(),text,horizon=32,beam_width=4,**protocol['arms']['full_w6'])
  archive=BASE/f'run/rank{index%8}/dev_{index:03d}.npz';old=read(BASE/'run/complete.json')['evaluation'][index]
  assert sha(archive)==old['artifact_sha256']
  with np.load(archive,allow_pickle=False) as z:
   assert np.array_equal(result['tokens'],z['tokens']),'Archived tokens differ'
   error=float(np.max(np.abs(result['prediction']-z['prediction'])));assert error<=1e-6,error
  scored=evaluator(result['prediction'],im[2:],[32])
  assert all(np.allclose(scored['per_frame'][k],old['per_frame'][k],atol=1e-5,rtol=0) for k in scored['per_frame'])
  del result
  longrow=protocol['cohorts']['extended_dev128']['cases'][a.rank]
  status(state='VERIFYING_H128_NATIVE_PREFIX',cases=0)
  rec=run_case('extended_dev128','full_w6',longrow,check_native_prefix=True)
  assert digest(actor)==protocol['checkpoint_digest']
  write(R/'parity.json',dict(state='PASS',protocol_sha256=protocol_sha,archived_index=index,exact_tokens=True,max_pixel_abs_error=error,
   long_case_id=longrow['id'],H128_native_H32_prefix=rec['native_H32_prefix_parity'],timestamp=time.time()))
  status(state='PARITY_PASS',cases=1);print('ALL_GPU_PARITY_PASS',a.rank,flush=True)
 else:
  for rank in range(4):
   gate=read(a.output/f'workers/rank{rank}/parity.json');assert gate['state']=='PASS' and gate['protocol_sha256']==protocol_sha
  tasks=[]
  # Rotate methods inside each case; complete all variants on all predeclared inputs.
  for cohort in ['extended_dev128','heldout_test128','historical_dev64']:
   for row in protocol['cohorts'][cohort]['cases']:
    for arm in ['full_w6','recent_w6','current_only','full_w1','full_w3','full_w12']:tasks.append((cohort,arm,row))
  own=[t for i,t in enumerate(tasks) if i%4==a.rank];start=time.monotonic()
  for n,(cohort,arm,row) in enumerate(own):
   status(state='EVALUATING',cases=n,total_cases=len(own),cohort=cohort,arm=arm,index=row['index'])
   run_case(cohort,arm,row)
  assert digest(actor)==protocol['checkpoint_digest']
  write(R/'complete.json',dict(state='SUCCEEDED',cases=len(own),protocol_sha256=protocol_sha,trained_updates=0,wall_seconds=time.monotonic()-start))
  status(state='SUCCEEDED',cases=len(own),total_cases=len(own))
except BaseException:
 write(R/'ERROR.json',dict(error=traceback.format_exc(),timestamp=time.time()));status(state='FAILED');raise
