from distributed import *
from futureworlds.evaluation import Metrics
import time
TEXT_MODE=P.get('text_mode','full');assert TEXT_MODE in ['full','no_text','shuffled']
PROTOCOL_SHA256=sha(E/'protocol.json')
EXPECTED_INITIAL=P['expected_initial_digest'];assert isinstance(EXPECTED_INITIAL,str) and len(EXPECTED_INITIAL)==64

def condition_record(row,used):
 donor=text_donor(row)
 return dict(text_mode=TEXT_MODE,original_instruction=row['text'],used_generator_instruction=used,
             generator_text_enabled=TEXT_MODE!='no_text',
             text_donor_id=donor['id'] if TEXT_MODE=='shuffled' else None,
             text_donor_cache_id=donor['cache_id'] if TEXT_MODE=='shuffled' else None,
             text_donor_map_sha256=P.get('text_donor_map_sha256') if TEXT_MODE=='shuffled' else None)
setup();source=verify_source();sel=json.loads((E/'selection.json').read_text());prepared=json.loads((E/'prepared.json').read_text());assert sha(E/'selection.json')==prepared['selection_sha256']
rows=sel['dev'];assert len(rows)==128
pipe=load_initial();actor=pipe.model.eval().requires_grad_(False);initial=agree(actor)
assert initial==EXPECTED_INITIAL,'Initializer differs from the pinned SFT/R2 checkpoint'
metric=Metrics('cuda:0');O=E/'evaluation';R=O/f'rank{RANK}';R.mkdir(parents=True,exist_ok=True);records=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
for i,row in enumerate(rows):
 if i%WORLD!=RANK:continue
 used=bind_generator_condition(pipe,row);conditioning=condition_record(row,used)
 path=R/f'dev_{i:03d}.json'
 if path.exists():
  rec=json.loads(path.read_text());assert rec['id']==row['id'] and rec['prepared_input_sha256']==row['sha256'];assert rec['initial_digest']==initial;assert rec['generator_condition']==conditioning and rec['protocol_sha256']==PROTOCOL_SHA256;assert sha(R/f'dev_{i:03d}.npz')==rec['artifact_sha256'];records.append(rec);continue
 setup(20261011+10000+i);images,actions,text=pixels(row)
 # Keep the judge/GT task unchanged; only the bound generator condition differs.
 used=bind_generator_condition(pipe,row);assert condition_record(row,used)==conditioning
 tick=time.monotonic();pred=pipe.predict(images[:2].copy(),actions[:33].copy(),used,horizon=32,beam_width=4,memory='full');torch.cuda.synchronize();infer=time.monotonic()-tick
 prediction=pred['prediction'];gt=images[2:34];scored=metric(prediction,gt,[10,20,32])
 if not records:
  identity=metric(gt.astype('float32')/255,gt,[32])['metrics']['32'];assert abs(identity['ssim']-1)<1e-5 and abs(identity['lpips'])<1e-5
 dest=R/f'dev_{i:03d}.npz';np.savez_compressed(dest,prediction=prediction,tokens=pred['tokens'],ground_truth=gt,observations=images[:2],instruction=np.array(text),used_instruction=np.array(used))
 rec=dict(id=row['id'],index=i,generator_condition=conditioning,protocol_sha256=PROTOCOL_SHA256,metrics=scored['metrics'],per_frame=scored['per_frame'],frame32={k:v[-1] for k,v in scored['per_frame'].items()},seconds=time.monotonic()-tick,inference_seconds=infer,initial_digest=initial,artifact_sha256=sha(dest),prepared_input_sha256=row['sha256'])
 write(path,rec);records.append(rec);write(R/'status.json',dict(state='EVALUATING',cases=len(records),total_cases=16));print('EVAL',RANK,len(records),flush=True);pipe.text_cache.clear()
assert digest(actor)==initial and len(records)==16;parts=gather(records);peak=gather(float(torch.cuda.max_memory_allocated()/2**30))
if RANK==0:
 rs=sorted([v for part in parts for v in part],key=lambda v:v['index']);assert [v['index'] for v in rs]==list(range(128))
 write(E/'complete.json',dict(state='SUCCEEDED',cases=128,updates=0,text_mode=TEXT_MODE,protocol_sha256=PROTOCOL_SHA256,source=source,initial_digest=initial,final_digest=initial,evaluation=rs,mean_frames1_32={k:float(np.mean([r['metrics']['32'][k] for r in rs])) for k in rs[0]['metrics']['32']},seconds=time.monotonic()-started,peak_allocated_GiB=max(peak),protocol=P))
dist.barrier();dist.destroy_process_group()
