"""Fixed cohort evaluation; never chooses models or reads scores to select cases."""
from distributed import *
import gc,time
from futureworlds.evaluation import Metrics
spec=json.loads(Path(os.environ['BRIDGE_EVAL_SPEC']).read_text())
O=Path(spec['output']);R=O/f'rank{RANK}'
if RANK==0:O.mkdir(parents=True,exist_ok=False)
dist.barrier();R.mkdir(exist_ok=False)
setup();verify_source();pipe=load_initial();initial=agree(pipe.model)
assert initial==P['expected_initial_digest']
cp=spec.get('checkpoint')
if cp:
 cp=Path(cp)
 for n in ['model.safetensors','text_adapters.safetensors']:assert sha(cp/n)==spec['checkpoint_files'][n]
 pipe.model.core.load_state_dict(load_file(str(cp/'model.safetensors'),device='cuda:0'),strict=True)
 pipe.model.adapters.load_state_dict(load_file(str(cp/'text_adapters.safetensors'),device='cuda:0'),strict=True)
final=agree(pipe.model);pipe.model.eval().requires_grad_(False)
metric=Metrics('cuda:0');rows=json.loads(Path(spec['cohort_file']).read_text());assert sha(spec['cohort_file'])==spec['cohort_sha256']
results=[];start=time.monotonic()
for i,row in enumerate(rows):
 if i%WORLD!=RANK:continue
 setup(20261011+10000+i)
 assert sha(row['path'])==row['sha256']
 with np.load(row['path'],allow_pickle=False) as z:
  images=z['image'][:34].copy();actions=z['action'][:34].astype('float32').copy()
 assert images.shape==(34,256,320,3) and actions.shape==(34,13)
 tick=time.monotonic();r=pipe.predict(images[:2].copy(),actions[:33].copy(),row['text'],horizon=32,beam_width=4,memory='full')
 pred=np.asarray(r['prediction'],dtype='float32');assert pred.shape==(32,256,320,3) and np.isfinite(pred).all()
 torch.cuda.synchronize();infer=time.monotonic()-tick
 scores=metric(pred,images[2:34],[10,20,32])
 if not results:
  ident=metric(images[2:34].astype('float32')/255,images[2:34],[32])['metrics']['32'];assert abs(ident['ssim']-1)<1e-5 and abs(ident['lpips'])<1e-5
 dest=R/f'case_{i:03d}.npz';np.savez_compressed(dest,prediction=pred,tokens=r['tokens'],ground_truth=images[2:34],observations=images[:2],instruction=np.array(row['text']))
 rec=dict(index=i,id=row['id'],input_sha256=row['sha256'],metrics=scores['metrics'],per_frame=scores['per_frame'],inference_seconds=infer,artifact=str(dest),artifact_sha256=sha(dest))
 write(R/f'case_{i:03d}.json',rec);results.append(rec);write(R/'status.json',dict(state='EVALUATING',completed=len(results),total=len(rows)//4));print('EVAL',RANK,len(results),flush=True)
assert len(results)==len(rows)//4 and digest(pipe.model)==final
parts=gather(dict(rows=results,seconds=time.monotonic()-start))
if RANK==0:
 result=sorted([r for p in parts for r in p['rows']],key=lambda r:r['index'])
 assert [r['id'] for r in result]==[r['id'] for r in rows] and len(set(r['id'] for r in result))==len(rows)
 means={m:float(np.mean([r['metrics']['32'][m] for r in result])) for m in result[0]['metrics']['32']}
 write(O/'complete.json',dict(state='SUCCEEDED',n=len(rows),mean_frames1_32=means,evaluation=result,initial_digest=initial,final_digest=final,spec=spec,eval_seconds=max(p['seconds'] for p in parts)))
 print('EVAL_COMPLETE',spec['name'],json.dumps(means),flush=True)
dist.barrier();dist.destroy_process_group()
