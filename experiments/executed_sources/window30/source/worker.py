"""Matched-horizon evaluation with independent timing and deterministic repeats."""
import argparse,fcntl,gc,sys,time,traceback
from common_new import *
a=argparse.ArgumentParser();a.add_argument('--rank',type=int,required=True);a.add_argument('--score',action='store_true');a=a.parse_args();assert 0<=a.rank<4
R=ROOT/f'workers/rank{a.rank}';R.mkdir(parents=True,exist_ok=True)
lock=(R/('scoring.lock' if a.score else 'generation.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=check();ph=sha(ROOT/'protocol.json');sys.path.insert(0,str(BASE))
from common import load_initial,setup,digest
import numpy as np
import torch
from safetensors.torch import load_file
from profile_decoder import predict_profiled
from window_decoder import validate_trace
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def data(row):
 assert sha(row['path'])==row['sha256']
 with np.load(row['path'],allow_pickle=False) as z:return z['image'].copy(),z['action'].astype('float32').copy(),str(z['instruction'])
def score_video(evaluator,pred,truth):
 assert len(pred)==len(truth) and len(pred)%16==0
 vectors={}
 for s in range(0,len(pred),16):
  block=evaluator(pred[s:s+16],truth[s:s+16],[16])
  for k,v in block['per_frame'].items():
   assert len(v)==16 and np.isfinite(v).all();vectors.setdefault(k,[]).extend(v)
 return dict(per_frame=vectors,metrics={k:float(np.mean(v)) for k,v in vectors.items()})
def status(state,**kw):write(R/('scoring_status.json' if a.score else 'generation_status.json'),dict(state=state,time=time.time(),**kw))
def dest(H,arm,i,rep):return ROOT/f'generation/h{H}/{arm}/case_{i:03d}_r{rep}.json'
def score_phase():
 from futureworlds.evaluation import Metrics
 setup(20261026);evaluator=Metrics('cuda:0')
 # Check new fixed16 metric batching against the previously verified full32 record.
 old=read(BASE/'run/complete.json')['evaluation'][a.rank];row=read(BASE/'selection.json')['dev'][a.rank]
 ap=BASE/f'run/rank{a.rank}/dev_{a.rank:03d}.npz';assert sha(ap)==old['artifact_sha256']
 with np.load(ap,allow_pickle=False) as z:pred=z['prediction'].copy()
 pred=np.ascontiguousarray(pred.transpose(0,3,1,2)).transpose(0,2,3,1)
 im,ac,tx=data(row);sc=score_video(evaluator,pred,im[2:34])
 errors={k:float(np.max(np.abs(np.asarray(v)-old['per_frame'][k]))) for k,v in sc['per_frame'].items()}
 assert max(errors.values())<=1e-5,errors
 write(R/'metric_gate.json',dict(state='PASS',protocol_sha256=ph,errors=errors,batch_size=16))
 done=0
 for row in p['cases']:
  if row['index']%4!=a.rank:continue
  im,ac,tx=data(row)
  for H in HORIZONS:
   for arm in ARMS:
    src=dest(H,arm,row['index'],0);rec=read(src)
    assert rec['protocol_sha256']==ph and rec['input_sha256']==row['sha256'] and rec['id']==row['id']
    ap=rec['artifact']['path'];assert sha(ap)==rec['artifact']['sha256']
    with np.load(ap,allow_pickle=False) as z:pred=z['prediction'].copy()
    pred=np.ascontiguousarray(pred.transpose(0,3,1,2)).transpose(0,2,3,1)
    scored=score_video(evaluator,pred,im[2:H+2])
    path=ROOT/f"quality/h{H}/{arm}/case_{row['index']:03d}.json"
    output=dict(index=row['index'],id=row['id'],horizon=H,arm=arm,input_sha256=row['sha256'],protocol_sha256=ph,generation_sha256=sha(src),**scored)
    if path.exists():assert read(path)==output
    else:write(path,output)
    done+=1;status('SCORING',completed=done);print('QUALITY',a.rank,done,flush=True)
 write(R/'scoring_complete.json',dict(state='SUCCEEDED',records=done,protocol_sha256=ph))
def generate():
 setup(20261026);status('LOADING');pipe=load_initial();cp=Path(p['checkpoint']);receipt=read(cp/'receipt.json')
 for n,target in [('model.safetensors',pipe.model.core),('text_adapters.safetensors',pipe.model.adapters)]:
  assert sha(cp/n)==receipt['files'][n];target.load_state_dict(load_file(str(cp/n),device='cuda:0'),strict=True)
 actor=pipe.model.eval().requires_grad_(False);assert digest(actor)==p['checkpoint_digest']
 assert all(v.dtype==torch.float32 for v in actor.parameters())
 def run(row,im,ac,tx,H,arm):
  W,anchor=ARMS[arm];pipe.text_cache.clear();pipe.condition(tx);setup(row['seed']);gc.collect()
  result,profile=predict_profiled(pipe,im[:2].copy(),ac[:H+1].copy(),tx,horizon=H,beam_width=4,recent_states=W,retain_anchor=anchor)
  assert result['tokens'].shape==(1,H,80) and result['prediction'].shape==(H,256,320,3)
  assert np.isfinite(result['prediction']).all();validate_trace(result['trace'],horizon=H,recent_states=W,retain_anchor=anchor)
  return result,profile
 # Instrumented decoder must reproduce the historical W6/H32 output exactly.
 old=read(BASE/'selection.json')['dev'][a.rank];old=dict(old,seed=20261011+10000+a.rank)
 im,ac,tx=data(old);res,prof=run(old,im,ac,tx,32,'w6');ap=BASE/f'run/rank{a.rank}/dev_{a.rank:03d}.npz'
 original=read(BASE/'run/complete.json')['evaluation'][a.rank];assert sha(ap)==original['artifact_sha256']
 with np.load(ap,allow_pickle=False) as z:
  assert np.array_equal(res['tokens'],z['tokens']);err=float(np.max(np.abs(res['prediction']-z['prediction'])))
 assert err<=1e-6;write(R/'generation_gate.json',dict(state='PASS',exact_tokens=True,max_pixel_error=err,protocol_sha256=ph))
 del res,prof;gc.collect()
 rows=[r for r in p['cases'] if r['index']%4==a.rank];warm=rows[0];im,ac,tx=data(warm)
 write(R/'device.json',dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],name=torch.cuda.get_device_name(),torch_version=torch.__version__))
 for H in HORIZONS:
  for arm in ARMS:
   status('WARMUP',horizon=H,arm=arm);res,prof=run(warm,im,ac,tx,H,arm)
   write(R/f'warm_h{H}_{arm}.json',dict(state='PASS',protocol_sha256=ph,profile=prof,included_in_statistics=False));del res,prof;gc.collect()
 done=0;arms=list(ARMS)
 for row in rows:
  im,ac,tx=data(row)
  for rep in range(3):
   hk=(row['index']+rep)%5;hs=HORIZONS[hk:]+HORIZONS[:hk]
   for H in hs:
    k=(row['index']+rep+HORIZONS.index(H))%5;order=arms[k:]+arms[:k]
    for arm in order:
     path=dest(H,arm,row['index'],rep)
     if path.exists():
      rec=read(path);assert rec['protocol_sha256']==ph and rec['input_sha256']==row['sha256']
      if rec.get('artifact'):assert sha(rec['artifact']['path'])==rec['artifact']['sha256']
      done+=1;continue
     status('MEASURING',completed=done,index=row['index'],horizon=H,arm=arm,repeat=rep)
     res,prof=run(row,im,ac,tx,H,arm);ts=arrsha(res['tokens']);ps=arrsha(res['prediction']);artifact=None
     if rep==0:
      ap=path.with_suffix('.npz');ap.parent.mkdir(parents=True,exist_ok=True);tmp=ap.with_suffix('.partial')
      with tmp.open('wb') as f:np.savez_compressed(f,prediction=res['prediction'],tokens=res['tokens'])
      tmp.replace(ap);artifact=dict(path=str(ap),sha256=sha(ap))
     else:
      first=read(dest(H,arm,row['index'],0));assert ts==first['tokens_sha256']
      with np.load(first['artifact']['path'],allow_pickle=False) as z:assert np.max(np.abs(res['prediction']-z['prediction']))<=1e-6
     rec=dict(index=row['index'],id=row['id'],horizon=H,arm=arm,repeat=rep,rank=a.rank,
       protocol_sha256=ph,input_sha256=row['sha256'],checkpoint_digest=p['checkpoint_digest'],profile=prof,
       tokens_sha256=ts,prediction_sha256=ps,artifact=artifact,trace=res['trace'])
     write(path,rec);done+=1;print('GENERATION',a.rank,done,row['index'],H,arm,rep,flush=True)
     del res,prof;gc.collect()
 write(R/'generation_complete.json',dict(state='SUCCEEDED',records=done,protocol_sha256=ph))
try:
 torch.cuda.set_device(0)
 if a.score:score_phase()
 else:generate()
 status('SUCCEEDED')
except BaseException:
 write(R/('ERROR_scoring.json' if a.score else 'ERROR_generation.json'),dict(error=traceback.format_exc(),time=time.time()));raise
