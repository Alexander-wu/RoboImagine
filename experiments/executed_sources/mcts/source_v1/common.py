import os,json,hashlib,sys,random,copy
from pathlib import Path
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
D=Path('/opt/roboimagine-workspace')
ROOT=Path(__file__).resolve().parent
E=Path(os.environ['R2_RUN_DIR']);LEGACY=D/'experiments/roboalign_r2_text_20261010'
sys.path.insert(0,str(ROOT/'code/src'))
C=D/'experiments/infinitystar_robot_codec_20261008'
sys.path.insert(0,str(C/'deps'));sys.path.insert(0,str(LEGACY/'reward_compare_v1/eval_deps'))
os.environ['TORCH_HOME']=str(C/'torch_cache')
import numpy as np,torch
from futureworlds.pipeline import WorldPipeline,encode_actions
from futureworlds.checkpoints import file_sha256 as sha
from futureworlds.data import write_json as write
from futureworlds.batch import TensorBatch
from futureworlds.training import policy_backward
from mcts_search import rollout_mcts
from futureworlds.search import rollout_group_beam
from futureworlds.ordinary_search import rollout_ordinary_beam
from futureworlds.memory import rollout,make_prefix
from safetensors.torch import load_file
import lpips
P=json.loads((E/'protocol.json').read_text())
TEXT_MODES={'full','no_text','shuffled'}
assert P.get('text_mode','full') in TEXT_MODES
def setup(seed=20261011):
 torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def verify_assets():
 for n,h in P['verified_files'].items():assert sha(n)==h,n

def load_initial():
 verify_assets();ds=P['dataset']
 p=WorldPipeline(D/'bundle',ds,'sft','cuda:0',str(D/'text_model'),verify=False)
 if ds=='bridge':
  from futureworlds.text_conditioning import TextWorldModel
  assert not p.settings['text_conditioning']['enabled']
  p.model=TextWorldModel(p.model).to('cuda:0').eval()
  p.settings['text_conditioning'].update(enabled=True,width=768,layers=[3,7,11],inner=256,max_tokens=256)
 if ds in ['rt1','bridge']:
  cp=Path(P.get('initial_checkpoint',D/'bundle'/ds/'models/sft'))
  # Both state dictionaries are required, including for evaluation checkpoints.
  # Asset hashes and the optional combined digest independently pin the identity.
  for name in ['model.safetensors','text_adapters.safetensors']:
   assert (cp/name).is_file(),str(cp/name)
   assert str(cp/name) in P['verified_files'],('Unpinned initializer',str(cp/name))
  p.model.core.load_state_dict(load_file(str(cp/'model.safetensors'),device='cuda:0'),strict=True)
  p.model.adapters.load_state_dict(load_file(str(cp/'text_adapters.safetensors'),device='cuda:0'),strict=True)
 assert p.settings['text_conditioning']['enabled'] and p.model.layers==[3,7,11]
 assert p.action_dim==P['action_dim']
 if P.get('expected_initial_digest') is not None:
  assert digest(p.model)==P['expected_initial_digest'],'Unexpected initial state digest'
 if P.get('text_mode','full')!='full':
  original=p.condition
  def condition(text):
   if P['text_mode']=='shuffled':
    row=getattr(p,'_r2_generator_row',None)
    assert row is not None,'Call bind_generator_condition(pipe,row) before shuffled prediction'
    used=generator_instruction(row)
    assert text in [row['text'],used],'Prediction text does not match the bound episode'
   else:used=text
   # The pipeline owns a text cache. Never change its tensors or dictionaries.
   c=_copy_tensors(original(used))
   if P['text_mode']=='no_text':c['text_mask']=torch.zeros_like(c['text_mask'])
   p.model.set_text(c['text_features'],c['text_mask']);return c
  p.condition=condition
 p.model.eval();return p

def pixels(row):
 assert sha(row['path'])==row['sha256']
 with np.load(row['path'],allow_pickle=False) as z:im=z['image'].copy();ac=z['action'].astype('float32').copy();text=str(z['instruction'])
 assert text==row['text'] and im.shape==(row['clip_length'],256,320,3) and im.dtype==np.uint8
 assert ac.shape==(len(im),P['action_dim']) and np.isfinite(ac).all()
 return im,ac,text

def _copy_tensors(value):
 if torch.is_tensor(value):return value.clone()
 if isinstance(value,dict):return {k:_copy_tensors(v) for k,v in value.items()}
 if isinstance(value,list):return [_copy_tensors(v) for v in value]
 if isinstance(value,tuple):return tuple(_copy_tensors(v) for v in value)
 return copy.deepcopy(value)

_DONORS=None
def _donor_lookup():
 global _DONORS
 if _DONORS is not None:return _DONORS
 path=Path(P['text_donor_map_file'])
 if not path.is_absolute():path=E/path
 assert sha(path)==P['text_donor_map_sha256'],'Text donor map changed'
 payload=json.loads(path.read_text());mapping=payload['mapping']
 selection=json.loads((E/'selection.json').read_text());lookup={};splits={}
 for split,rows in selection.items():
  for row in rows:
   key=row['cache_id'];assert key not in lookup,('Duplicate cache ID',key)
   lookup[key]=row;splits[key]=split
 assert set(mapping)==set(lookup),'Shuffled-text map must cover train, smoke and dev exactly'
 assert len(set(mapping.values()))==len(lookup),'Text donor map must be a bijection'
 for key,donor_key in mapping.items():
  assert key in lookup and donor_key in lookup,('Unknown donor/recipient',key,donor_key)
  recipient,donor=lookup[key],lookup[donor_key]
  assert splits[key]==splits[donor_key],('Cross-split text donor',key,donor_key)
  assert recipient['id']!=donor['id'] and recipient['text']!=donor['text'],('Non-intervening donor',key,donor_key)
 _DONORS={key:(lookup[key],lookup[donor_key]) for key,donor_key in mapping.items()}
 return _DONORS

def text_donor(row):
 if P.get('text_mode','full')!='shuffled':return row
 recipient,donor=_donor_lookup()[row['cache_id']]
 for key in ['id','text','path','sha256','cache_id','cache_sha256']:
  assert row[key]==recipient[key],('Recipient changed',key,row['id'])
 return donor

def generator_instruction(row):
 """The judge keeps row['text']; only the generator uses this instruction."""
 return text_donor(row)['text']

def bind_generator_condition(pipe,row):
 """Bind the episode before predict(); returns its generator instruction."""
 pipe._r2_generator_row=row
 return generator_instruction(row)

def _raw_cache(row):
 p=E/'cache'/f"{row['cache_id']}.pt";assert sha(p)==row['cache_sha256']
 return _copy_tensors(torch.load(p,map_location='cuda:0',weights_only=True))

def load_cache(row):
 c=_raw_cache(row);mode=P.get('text_mode','full')
 if mode=='no_text':c['mask']=torch.zeros_like(c['mask'])
 elif mode=='shuffled':
  donor=_raw_cache(text_donor(row));c['features']=donor['features'];c['mask']=donor['mask']
 return c

def cond(c):return dict(text_features=c['features'],text_mask=c['mask'])

def propose_candidates(model,context,anchor,actions,horizon,seed):
 config=P['search'];assert config['strategy']=='mcts'
 f,trace=rollout_mcts(model,context,anchor,actions,horizon,
  torch.Generator(device=context.device).manual_seed(seed),
  simulations=config['simulations'],branching=config['branching'],returned=4,
  c_uct=config['c_uct'],temperature=1.,recent=6)
 assert len(f)==horizon and all(frame['response'].shape==(4,80) for frame in f)
 trace.update(returned_trajectories=4,reward_based_selection=False,
  vocabulary_support=4375,training_horizon=horizon,seed=seed)
 return f,trace

def candidates(p,row,seed,horizon=10):
 bind_generator_condition(p,row)
 c=load_cache(row);p.model.set_text(c['features'],c['mask'])
 f,trace=propose_candidates(p.model,c['context'].reshape(1,-1)+4375,c['dynamics'][1:2],c['actions'],horizon,seed)
 codes=torch.stack([x['response'] for x in f],1)
 with torch.no_grad():pred=p.worker.detokenize(TensorBatch({'tokens':codes,'ctx_tokens':c['context'].expand(4,-1,-1)}),None).batch['pixels'][:,1:]
 assert pred.shape==(4,horizon,3,256,320) and torch.isfinite(pred).all()
 return f,trace,pred,c
class Metric:
 def __init__(self):self.lp=lpips.LPIPS(net='vgg').to('cuda:0').eval().requires_grad_(False)
 @torch.no_grad()
 def __call__(self,pred,gt):
  gt=torch.as_tensor(gt,device='cuda:0').permute(0,3,1,2).float()/255;out=[]
  for p in pred:
   mae=(p-gt).abs().mean();lps=torch.cat([self.lp(p[i:i+2]*2-1,gt[i:i+2]*2-1).flatten() for i in range(0,len(gt),2)]).mean()
   out.append(dict(mae=float(mae),lpips=float(lps),reward=-float(mae+lps)))
  return out

def mix(m,s):
 assert m.shape==s.shape and torch.isfinite(m).all() and torch.isfinite(s).all()
 return .75*(m-m.mean())/m.std().clamp_min(1e-6)+.25*(s-s.mean())/s.std().clamp_min(1e-6)
def as_video(pred):return np.round(pred.permute(0,2,3,1).detach().cpu().clamp(0,1).numpy()*255).astype('uint8')
