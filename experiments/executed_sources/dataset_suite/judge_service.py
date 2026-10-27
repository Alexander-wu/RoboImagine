import os,sys,json,time,hashlib,inspect,traceback
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
D=Path('/opt/roboimagine-workspace')
E=D/'experiments/roboalign_r2_text_20261010';M=E/'judge_mjev_v1';V=E/'judge_instruct_v4'
sys.path.insert(0,str(Path(__file__).resolve().parent));from judge_prompt_snapshot import ABS,PAIR
ARM=sys.argv[1];OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
assert ARM in ['base4b','mjev4b']
def atomic(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x: sorted(x) if isinstance(x,set) else str(x)));tmp.replace(p)
manifest=json.loads((M/('model_manifest.json' if ARM=='mjev4b' else 'base4b/model_manifest.json')).read_text());assert manifest['state']=='READY'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
model,info=Qwen3VLForConditionalGeneration.from_pretrained(manifest['path'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
model.eval().requires_grad_(False)
assert __import__('transformers').__version__=='5.14.1'
processor=AutoProcessor.from_pretrained(manifest['path'],local_files_only=True)
PROTOCOL={'records_per_model':7800}  # upper bound:650groups x12ordered pair comparisons
atomic(OUT/'loading.json',dict(arm=ARM,manifest=manifest,loading_info=info,transformers=__import__('transformers').__version__))
ABS_DESCS=['clearly task-incompatible visible behavior','no observable task progress','observable compatible partial progress','a clearly achieved instructed visible subgoal','insufficient evidence']
PAIR_DESCS=['Clip A has more visible task-compatible progress','Clip B has more visible task-compatible progress','no meaningful visible difference in task progress','insufficient evidence to compare']
def witness():
 h=hashlib.sha256()
 for n,p in model.named_parameters():
  h.update(n.encode());h.update(p.detach().flatten()[:16].float().cpu().numpy().tobytes())
 return h.hexdigest()
before=witness();records=[];started=time.monotonic();checks=[]
def content(initial,clips):
 out=[{'type':'text','text':'Shared initial observation, BEFORE either clip.'},{'type':'image','image':Image.fromarray(initial)}]
 for name,video in clips:
  assert video.dtype==np.uint8 and video.shape==(10,256,320,3)
  out.append({'type':'text','text':f'BEGIN CLIP {name}: eight frames in chronological display order.'})
  for t in np.linspace(0,9,8).round().astype(int):
   out.extend([{'type':'text','text':f'Clip {name}, frame +{t+1}'},{'type':'image','image':Image.fromarray(video[t])}])
  out.append({'type':'text','text':f'END CLIP {name}'})
 return out
@torch.inference_mode()
def run(initial,clips,instruction,identifier,mode,layout,check=False,repeat=False):
 torch.cuda.synchronize();tick=time.monotonic()
 semantics=list('0123U') if mode=='absolute' else list('ABTU')
 prompt=(ABS if mode=='absolute' else PAIR)+instruction
 labels=semantics
 if layout=='candidate':
  # Reproduce upstream causal candidate readout grammar without its experimental cache/mask engine.
  scope=prompt.split('Choose exactly one label:')[0]
  question='Which description best matches the visible task progress?' if mode=='absolute' else 'Which clip shows more visible task-compatible progress?'
  descriptions=ABS_DESCS if mode=='absolute' else PAIR_DESCS
  labels=list('ABCDE')[:len(semantics)]
  prompt=scope+'\nTask: '+instruction+'\nQuestion: '+question+'\nChoose exactly one candidate and answer with its label.\nCandidates:\n'+''.join(f'[{label}] {desc}\n' for label,desc in zip(labels,descriptions))
 c=content(initial,clips);c.append({'type':'text','text':prompt});messages=[{'role':'user','content':c}]
 rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 if layout=='candidate':rendered+='Answer:\n'
 inputs=processor(text=[rendered],images=[x['image'] for x in c if x['type']=='image'],return_tensors='pt',padding=False,truncation=False)
 raw_ids=processor.tokenizer.encode(rendered,add_special_tokens=False);label_ids=[]
 for label in labels:
  joint=processor.tokenizer.encode(rendered+label,add_special_tokens=False)
  assert len(joint)==len(raw_ids)+1 and joint[:-1]==raw_ids,(layout,label)
  label_ids.append(joint[-1])
 assert len(set(label_ids))==len(labels)
 if check and layout=='short':
  native=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
  for key in ['input_ids','attention_mask','pixel_values','image_grid_thw']:
   assert torch.equal(inputs[key],native[key]),f'Historical input mismatch: {key}'
 inputhash=hashlib.sha256(initial.tobytes()+instruction.encode())
 for name,video in clips:inputhash.update(name.encode());inputhash.update(video.tobytes())
 token_sha=hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest()
 pixel_sha=hashlib.sha256(inputs['pixel_values'].numpy().tobytes()).hexdigest()
 tokens=inputs['input_ids'].shape[1];assert tokens<12000,'Never truncate long input'
 inputs=inputs.to(model.device)
 torch.cuda.synchronize();gpu_start=time.monotonic()
 output=model(**inputs,logits_to_keep=1,use_cache=False,return_dict=True)
 logits=output.logits[0,-1,label_ids].float();prob=logits.softmax(-1);torch.cuda.synchronize();end=time.monotonic()
 probs={s:float(v) for s,v in zip(semantics,prob.cpu().tolist())}
 raw=semantics[int(prob.argmax())]
 rec=dict(id=identifier,mode=mode,layout=layout,arm=ARM,raw=raw,label=labels[int(prob.argmax())],probabilities=probs,logits=dict(zip(semantics,logits.cpu().tolist())),label_token_ids=label_ids,
  seconds=end-tick,gpu_seconds=end-gpu_start,input_tokens=tokens,input_sha256=inputhash.hexdigest(),token_sha256=token_sha,pixel_sha256=pixel_sha,prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),errors=[],repeat=repeat)
 if mode=='absolute':
  visible_mass=1-probs['U'];rec['visible_mass']=visible_mass
  rec['conditional_progress']=sum(int(k)*v for k,v in probs.items() if k!='U')/(3*visible_mass) if visible_mass>1e-8 else None
  rec['progress_mass']=sum(int(k)*v for k,v in probs.items() if k!='U')/3
 # Correctness witness only; not included in latency measurement.
 if check:
  def allowed(batch,ids):return label_ids
  y=model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=1,prefix_allowed_tokens_fn=allowed)
  gen_id=int(y[0,-1]);assert gen_id==label_ids[int(prob.argmax())],('Readout/generation mismatch',gen_id,rec)
  checks.append(dict(id=identifier,layout=layout,direct_token=label_ids[int(prob.argmax())],generated_token=gen_id,pass_check=True))
 atomic(OUT/(identifier+'.json'),rec)
 if not identifier.startswith('warmup'):records.append(rec)
 atomic(OUT/'progress.json',dict(completed=len(records),expected=PROTOCOL['records_per_model'],last=identifier,elapsed_seconds=time.monotonic()-started))
 print('JUDGMENT',json.dumps({k:rec[k] for k in ['id','raw','seconds','probabilities']}),flush=True)
 return rec
atomic(OUT/'READY.json',dict(state='READY',pid=os.getpid(),witness=before))
try:
 for line in sys.stdin:
  req=json.loads(line);path=Path(req['path']);z=np.load(path,allow_pickle=False)
  assert z['videos'].shape==(4,10,256,320,3)
  scores=np.zeros(4,dtype='float64');pairs=[]
  for i in range(4):
   for j in range(i+1,4):
    prefix=f"g{req['step']:04d}_p{i}{j}"
    f=run(z['initial'],[('A',z['videos'][i]),('B',z['videos'][j])],str(z['instruction']),prefix+'_f','pair','candidate',check=req['step']==1 and i==0 and j==1)
    b=run(z['initial'],[('A',z['videos'][j]),('B',z['videos'][i])],str(z['instruction']),prefix+'_b','pair','candidate')
    pf=f['probabilities'];pb=b['probabilities'];d=.5*((pf['A']-pf['B'])-(pb['A']-pb['B']))
    scores[i]+=d/3;scores[j]-=d/3
    pairs.append(dict(i=i,j=j,margin=d,forward=pf,reverse=pb,seconds=f['seconds']+b['seconds']))
  assert np.isfinite(scores).all() and abs(scores.sum())<1e-8
  atomic(Path(req['result']),dict(scores=scores.tolist(),pairs=pairs,judge_seconds=sum(p['seconds'] for p in pairs),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  z.close()
 assert witness()==before
 atomic(OUT/'complete.json',dict(state='SUCCEEDED',records=len(records),frozen_witness_unchanged=True,checks=checks))
except Exception:
 atomic(OUT/'ERROR.json',dict(error=traceback.format_exc()));raise
