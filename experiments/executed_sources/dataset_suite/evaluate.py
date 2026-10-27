from distributed import *
from futureworlds.evaluation import Metrics
import time
setup();verify_source();sel=json.loads((E/'selection.json').read_text());prepared=json.loads((E/'prepared.json').read_text());assert sha(E/'selection.json')==prepared['selection_sha256']
rows=sel['dev'];assert len(rows)==128
pipe=load_initial();actor=pipe.model.eval().requires_grad_(False);initial=agree(actor)
if P.get('text_mode')!='no_text':assert initial==prepared['initial_digest']
metric=Metrics('cuda:0');O=E/'evaluation';R=O/f'rank{RANK}';R.mkdir(parents=True,exist_ok=True);records=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
for i,row in enumerate(rows):
 if i%WORLD!=RANK:continue
 path=R/f'dev_{i:03d}.json'
 if path.exists():
  rec=json.loads(path.read_text());assert rec['id']==row['id'] and rec['prepared_input_sha256']==row['sha256'];assert rec['initial_digest']==initial;assert sha(R/f'dev_{i:03d}.npz')==rec['artifact_sha256'];records.append(rec);continue
 setup(20261011+10000+i);images,actions,text=pixels(row);used=text
 if P.get('text_mode')=='empty':used=''
 if P.get('text_mode')=='shuffled':
  used=next(rows[(i+j)%128]['text'] for j in range(1,128) if rows[(i+j)%128]['text']!=text)
 tick=time.monotonic();pred=pipe.predict(images[:2].copy(),actions[:33].copy(),used,horizon=32,beam_width=4,memory='full');torch.cuda.synchronize();infer=time.monotonic()-tick
 prediction=pred['prediction'];gt=images[2:34];scored=metric(prediction,gt,[10,20,32])
 if not records:
  identity=metric(gt.astype('float32')/255,gt,[32])['metrics']['32'];assert abs(identity['ssim']-1)<1e-5 and abs(identity['lpips'])<1e-5
 dest=R/f'dev_{i:03d}.npz';np.savez_compressed(dest,prediction=prediction,tokens=pred['tokens'],ground_truth=gt,observations=images[:2],instruction=np.array(text),used_instruction=np.array(used))
 rec=dict(id=row['id'],index=i,metrics=scored['metrics'],per_frame=scored['per_frame'],frame32={k:v[-1] for k,v in scored['per_frame'].items()},seconds=time.monotonic()-tick,inference_seconds=infer,initial_digest=initial,artifact_sha256=sha(dest),prepared_input_sha256=row['sha256'])
 write(path,rec);records.append(rec);write(R/'status.json',dict(state='EVALUATING',cases=len(records),total_cases=16));print('EVAL',RANK,len(records),flush=True);pipe.text_cache.clear()
assert digest(actor)==initial and len(records)==16;parts=gather(records);peak=gather(float(torch.cuda.max_memory_allocated()/2**30))
if RANK==0:
 rs=sorted([v for part in parts for v in part],key=lambda v:v['index']);assert [v['index'] for v in rs]==list(range(128))
 write(E/'complete.json',dict(state='SUCCEEDED',cases=128,updates=0,initial_digest=initial,final_digest=initial,evaluation=rs,mean_frames1_32={k:float(np.mean([r['metrics']['32'][k] for r in rs])) for k in rs[0]['metrics']['32']},seconds=time.monotonic()-started,peak_allocated_GiB=max(peak),protocol=P))
dist.barrier();dist.destroy_process_group()
