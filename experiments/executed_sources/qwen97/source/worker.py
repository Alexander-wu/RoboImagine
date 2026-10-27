import time,traceback,random,fcntl
import numpy as np,torch,transformers
from PIL import Image
from shared import *
rank=int(os.environ['EVALUATION_RANK']);scope=sealed();torch.set_num_threads(3);assert torch.cuda.device_count()==1
identity=json.loads((ROOT/'identity.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text());rows=manifest['rows']
assert manifest['identity']==identity['evaluation_id']
status=ROOT/f'rank{rank}/status.json';result_dir=ROOT/'judgments';result_dir.mkdir(exist_ok=True)
def progress(state,**kwargs):write(status,dict(state=state,rank=rank,pid=os.getpid(),time=time.time(),**kwargs))
torch.cuda.set_per_process_memory_fraction(32*2**30/torch.cuda.get_device_properties(0).total_memory)
progress('LOADING')
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL),dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
write(ROOT/f'rank{rank}/runtime.json',{'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'image_processor':processor.image_processor.to_dict(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})
def witness():
 return {n:hashlib.sha256(p.detach().flatten()[:64].float().cpu().numpy().tobytes()).hexdigest() for n,p in list(model.named_parameters())[::max(1,len(list(model.named_parameters()))//8)]}
frozen_before=witness();assert all(not p.requires_grad for p in model.parameters())
write(ROOT/f'rank{rank}/frozen_before.json',frozen_before)
completed=0;valid_count=0
@torch.inference_mode()
def evaluate(row):
 global completed,valid_count
 dest=result_dir/(row['uid']+'.json')
 if dest.exists():
  rec=json.loads(dest.read_text());assert rec['evaluation_id']==identity['evaluation_id'] and rec['artifact_sha256']==row['artifact_sha256'];return rec
 assert sha(row['path'])==row['artifact_sha256']
 with np.load(row['path'],allow_pickle=False) as n:
  initial=n['observations'][1].copy();instruction=str(n['instruction'].item())
  if row['kind']=='recorded':video=n['ground_truth'].copy()
  elif row['kind']=='static':video=np.repeat(initial[None],32,axis=0)
  else:video=np.rint(n['prediction'].clip(0,1)*255).astype(np.uint8)
 indices=PROTOCOL['sampled_indices'];h=hashlib.sha256(initial.tobytes()+video[indices].tobytes()+instruction.encode()).hexdigest()
 assert h==row['expected_judge_input_sha256']
 content=[{'type':'text','text':'Instruction: '+instruction+'\nInitial Frame (starting state):'},{'type':'image','image':Image.fromarray(initial)}, {'type':'text','text':'Generated Video Frames (8 frames sampled uniformly from a 32-frame clip, in chronological order):'}]
 for i in indices:content.extend([{'type':'text','text':f'Frame +{i+1}'},{'type':'image','image':Image.fromarray(video[i])}])
 content.append({'type':'text','text':'Please evaluate this generated video based on the 6 dimensions described.'})
 messages=[{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':content}]
 inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda:0')
 rec=dict(row,evaluation_id=identity['evaluation_id'],judge_input_sha256=h,prompt_sha256=PROTOCOL['prompt_sha256'],input_tokens=int(inputs['input_ids'].shape[1]),image_grid_thw=inputs.get('image_grid_thw').tolist(),attempts=[],valid=False)
 if row.get('prior_invalid_path'):
  assert sha(row['prior_invalid_path'])==row['prior_invalid_sha256']
  prior=json.loads(Path(row['prior_invalid_path']).read_text())
  rec['attempts']=prior['attempts']
  assert len(rec['attempts'])==2 and all(a.get('error') for a in rec['attempts'])
 for attempt in range(3):
  if attempt<len(rec['attempts']):continue
  generation=dict(PROTOCOL['generation'])
  if attempt==2:generation.update(max_new_tokens=8192,max_time=360.0)
  progress('SCORING',uid=row['uid'],attempt=attempt,completed=completed,valid=valid_count,total=len(rows))
  cached_attempt=ROOT/'attempts'/f"{row['uid']}_{attempt}.json"
  if cached_attempt.exists():
   prior=json.loads(cached_attempt.read_text());assert prior['evaluation_id']==identity['evaluation_id'] and prior['seed']==row['seed']+1000000*attempt
   a={k:v for k,v in prior.items() if k not in ['evaluation_id','uid']};rec['attempts'].append(a)
   if a.get('error') is None:
    rec['judgment']=a['judgment'];rec['valid']=True;break
   continue
  seed=row['seed']+1000000*attempt;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  torch.cuda.synchronize();tick=time.monotonic()
  y=model.generate(**inputs,**generation,use_cache=True)
  torch.cuda.synchronize();new=y[0,inputs['input_ids'].shape[1]:];raw=processor.decode(new,skip_special_tokens=True)
  eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
  a={'attempt':attempt,'seed':seed,'generation':generation,'extended_format_repair':attempt==2,'seconds':time.monotonic()-tick,'output_tokens':len(new),'raw_response':raw,'finished_eos':int(new[-1]) in eos,'error':None}
  try:
   a['judgment']=parse(raw,truncated=(not a['finished_eos'] or len(new)>=generation['max_new_tokens']))
   rec['judgment']=a['judgment'];rec['valid']=True
  except ValueError as e:a['error']=str(e)
  rec['attempts'].append(a);write(ROOT/'attempts'/f"{row['uid']}_{attempt}.json",dict(evaluation_id=identity['evaluation_id'],uid=row['uid'],**a))
  if rec['valid']:break
  del y,new
 rec['seconds']=sum(a['seconds'] for a in rec['attempts']);write(dest,rec)
 print(json.dumps({'event':'JUDGMENT','rank':rank,'uid':row['uid'],'valid':rec['valid'],'total':rec.get('judgment',{}).get('total'),'seconds':rec['seconds']},ensure_ascii=False),flush=True)
 return rec
def claim():
 with open(ROOT/'queue.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for row in rows:
   if (result_dir/(row['uid']+'.json')).exists():continue
   dest=ROOT/'claims'/(row['uid']+'.json')
   if dest.exists():continue
   write(dest,{'uid':row['uid'],'rank':rank,'pid':os.getpid(),'time':time.time()})
   return row
 return None
try:
 while True:
  row=claim()
  if row is None:break
  rec=evaluate(row);completed+=1;valid_count+=int(rec['valid']);progress('EVALUATING',completed=completed,valid=valid_count)
 assert witness()==frozen_before
 write(ROOT/f'rank{rank}/frozen_after.json',witness())
 progress('SUCCEEDED',completed=completed,valid=valid_count,frozen_witness_unchanged=True)
except BaseException:
 progress('FAILED',completed=completed,valid=valid_count,error=traceback.format_exc());raise
