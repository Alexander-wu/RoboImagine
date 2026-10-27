"""Full128 checkpoint evaluation. Same Figure8 inputs/decoder/seeds; no training."""
import os,sys,json,time,traceback,hashlib
from pathlib import Path
D=Path('/opt/roboimagine-workspace')
EXP=D/'experiments/roboimagine_mcts_rt1_20261025';SRC=EXP/'source_v1';CFG=EXP/'config_v1'
OUT=D/'experiments/roboimagine_search_curves_full128_20261026/v1';RANK=int(os.environ['EVAL_RANK'])
os.environ['R2_RUN_DIR']=str(CFG);sys.path.insert(0,str(SRC))
from common import *
from futureworlds.evaluation import Metrics
PIN=json.loads((OUT/'source/checkpoints.json').read_text())
ARMS=['mcts100','ordinary100'] if RANK==0 else ['mcts250','ordinary250']
PILOTS={'mcts100':'pilot100_v1','mcts250':'pilot250_v1','ordinary100':'beam_pilot100_v1'}
def emit(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)
def run():
 setup();torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory)
 for name,h in json.loads((SRC/'source_manifest.json').read_text()).items():assert sha(SRC/name)==h,name
 selection=json.loads((CFG/'selection.json').read_text())['dev'];assert len(selection)==128
 prepared=json.loads((CFG/'prepared.json').read_text());assert sha(CFG/'selection.json')==prepared['selection_sha256']
 base=json.loads(Path(P['baseline_complete']).read_text());assert base['initial_digest']==P['expected_initial_digest']
 assert all(row['id']==base['evaluation'][i]['id'] and row['sha256']==base['evaluation'][i]['prepared_input_sha256'] for i,row in enumerate(selection))
 emit(f'rank{RANK}/protocol.json',dict(indices=list(range(128)),ids=[r['id'] for r in selection],horizon=32,beam=4,memory='full',seed='20261011+10000+index',source_sha=sha(SRC/'source_manifest.json'),selection_sha=sha(CFG/'selection.json'),baseline_sha=sha(P['baseline_complete']),initial_digest=P['expected_initial_digest'],arms=ARMS))
 pipe=load_initial();pipe.model.eval().requires_grad_(False);metric=Metrics('cuda:0')
 def evaluate(i):
  setup(20261011+10000+i);row=selection[i];images,actions,text=pixels(row);used=bind_generator_condition(pipe,row)
  with torch.inference_mode():
   res=pipe.predict(images[:2].copy(),actions[:33].copy(),used,horizon=32,beam_width=4,memory='full')
   pred=np.asarray(res['prediction'],dtype='float32');gt=images[2:34];val=metric(pred,gt,[10,20,32])
  assert pred.shape==(32,256,320,3) and np.isfinite(pred).all()
  return res,pred,gt,images,text,val
 # Same runtime reproduces the recorded shared initial checkpoint before any curve points.
 res,pred,gt,images,text,val=evaluate(0)
 assert all(abs(val['metrics']['32'][k]-base['evaluation'][0]['metrics']['32'][k])<1e-4 for k in ['psnr','ssim','lpips'])
 emit(f'rank{RANK}/baseline_parity.json',dict(state='PASS',index=0,metrics=val,reference=base['evaluation'][0]['metrics']))
 for arm in ARMS:
  pin=PIN[arm];cp=Path(pin['path']);rec=json.loads((cp/'receipt.json').read_text())
  assert rec['updates']==pin['updates'] and rec['digest']==pin['digest'] and rec['initial_digest']==P['expected_initial_digest']
  assert rec['files']==pin['files']
  for name,h in rec['files'].items():assert sha(cp/name)==h,(arm,name)
  emit(f'{arm}/checkpoint_receipt.json',dict(path=str(cp),receipt_sha=sha(cp/'receipt.json'),receipt=rec))
  pipe.model.core.load_state_dict(load_file(str(cp/'model.safetensors'),device='cuda:0'),strict=True)
  pipe.model.adapters.load_state_dict(load_file(str(cp/'text_adapters.safetensors'),device='cuda:0'),strict=True)
  assert digest(pipe.model)==pin['digest']
  cached={}
  if arm in PILOTS:
   oldroot=EXP/PILOTS[arm];old=json.loads((oldroot/'complete.json').read_text());assert old['state']=='SUCCEEDED'
   for x in old['per_case'][arm]:
    i=x['index'];assert x['id']==selection[i]['id'] and x['prepared_input_sha256']==selection[i]['sha256']
    path=oldroot/f'{arm}_{i:03d}.npz';assert sha(path)==x['artifact_sha256']
    with np.load(path,allow_pickle=False) as z:
     assert z['prediction'].shape==(32,256,320,3) and np.isfinite(z['prediction']).all()
    cached[i]=dict(x,artifact_path=str(path),checkpoint_digest=pin['digest'],origin='verified_pilot',source_complete_sha=sha(oldroot/'complete.json'))
   assert len(cached)==8
   # Checkpoint/runtime parity against an archived prediction before reusing prior eight.
   res,pred,gt,images,text,val=evaluate(0)
   with np.load(cached[0]['artifact_path'],allow_pickle=False) as z:
    assert np.array_equal(res['tokens'],z['tokens']) and np.max(np.abs(pred-z['prediction']))<1e-5
   assert all(abs(val['metrics']['32'][k]-cached[0]['metrics']['32'][k])<1e-5 for k in ['psnr','ssim','lpips'])
   emit(f'{arm}/pilot_parity.json',{'state':'PASS','index':0,'tokens_exact':True})
  rows=[]
  for i,row in enumerate(selection):
   emit(f'rank{RANK}/status.json',dict(state='EVALUATING',arm=arm,index=i,completed=len(rows),time=time.time()))
   dest=OUT/f'{arm}/case_{i:03d}.json'
   if dest.exists():
    x=json.loads(dest.read_text());assert x['id']==row['id'] and x['checkpoint_digest']==pin['digest'] and sha(x['artifact_path'])==x['artifact_sha256']
   elif i in cached:x=cached[i];emit(f'{arm}/case_{i:03d}.json',x)
   else:
    res,pred,gt,images,text,val=evaluate(i)
    path=OUT/f'{arm}/prediction_{i:03d}.npz';temp=path.with_suffix('.partial')
    with temp.open('wb') as f:np.savez_compressed(f,prediction=pred,tokens=res['tokens'],ground_truth=gt,observations=images[:2],instruction=np.array(text))
    temp.replace(path)
    x=dict(index=i,id=row['id'],prepared_input_sha256=row['sha256'],artifact_path=str(path),artifact_sha256=sha(path),checkpoint_digest=pin['digest'],origin='new_full128',**val)
    emit(f'{arm}/case_{i:03d}.json',x)
   assert x['prepared_input_sha256']==row['sha256'];rows.append(x)
   print('CASE',arm,len(rows),128,x['origin'],x['metrics']['32']['psnr'],flush=True)
  assert digest(pipe.model)==pin['digest']
  means={k:float(np.mean([r['metrics']['32'][k] for r in rows])) for k in rows[0]['metrics']['32']}
  emit(f'{arm}/complete.json',dict(state='SUCCEEDED',cases=128,updates=pin['updates'],metrics=means,per_case=rows,checkpoint_digest=pin['digest'],finished=time.time()))
 emit(f'rank{RANK}/status.json',dict(state='SUCCEEDED',time=time.time(),arms=ARMS))
if __name__=='__main__':
 try:run()
 except BaseException:
  emit(f'rank{RANK}/ERROR.json',{'error':traceback.format_exc(),'time':time.time()});raise
