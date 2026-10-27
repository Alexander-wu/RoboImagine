import os,json,hashlib,sys,random
from pathlib import Path
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
D=Path('/opt/roboimagine-workspace')
E=D/'experiments/rt1_mjev_20261011/sft_mjev500_8gpu_v1'
LEGACY=D/'experiments/roboalign_r2_text_20261010'
sys.path.insert(0,str(E/'code/src'))
C=D/'experiments/infinitystar_robot_codec_20261008'
sys.path.insert(0,str(C/'deps'));sys.path.insert(0,str(LEGACY/'reward_compare_v1/eval_deps'))
os.environ['TORCH_HOME']=str(C/'torch_cache')
import numpy as np,torch
from futureworlds.pipeline import WorldPipeline
from futureworlds.checkpoints import file_sha256 as sha
from futureworlds.data import write_json as write
from futureworlds.batch import TensorBatch
from futureworlds.training import policy_backward
from futureworlds.search import rollout_group_beam
import lpips
P=json.loads((E/'protocol.json').read_text())
def setup(seed=20261011):
 torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def load_initial():
 for name,v in P['assets'].items():
  if not name.startswith('rt1/models/main/'):
   assert sha(D/'bundle'/name)==v['sha256'],name
 p=WorldPipeline(D/'bundle','rt1','sft','cuda:0',str(D/'text_model'),verify=False)
 assert p.settings['text_conditioning']['enabled'] and p.model.layers==[3,7,11]
 assert p.settings['variants']['sft']['updates']==50000 and p.settings['variants']['sft']['method']=='SFT'
 p.model.eval();return p

def pixels(row):
 assert sha(row['path'])==row['sha256']
 with np.load(row['path'],allow_pickle=False) as z:im=z['image'].copy();ac=z['action'].astype('float32').copy();text=str(z['instruction'])
 assert text==row['text'] and im.shape==(row['clip_length'],256,320,3) and im.dtype==np.uint8
 assert ac.shape==(len(im),13) and np.isfinite(ac).all()
 return im,ac,text

def load_cache(row):
 p=E/'cache'/f"{row['cache_id']}.pt";assert sha(p)==row['cache_sha256']
 return torch.load(p,map_location='cuda:0',weights_only=True)
def cond(c):return dict(text_features=c['features'],text_mask=c['mask'])
def candidates(p,row,seed,horizon=10):
 c=load_cache(row);p.model.set_text(c['features'],c['mask'])
 f,trace=rollout_group_beam(p.model,c['context'].reshape(1,-1)+4375,c['dynamics'][1:2],c['actions'],horizon,penalty=.1,groups=4,beams=2)
 codes=torch.stack([x['response'] for x in f],1)
 with torch.no_grad():pred=p.worker.detokenize(TensorBatch({'tokens':codes,'ctx_tokens':c['context'].expand(4,-1,-1)}),None).batch['pixels'][:,1:]
 assert pred.shape==(4,horizon,3,256,320) and torch.isfinite(pred).all()
 return f,trace,pred,c
class Metric:
 def __init__(self):self.lp=lpips.LPIPS(net='vgg').to('cuda:0').eval().requires_grad_(False)
 @torch.no_grad()
 def __call__(self,pred,gt):
  gt=torch.as_tensor(gt,device='cuda:0').permute(0,3,1,2).float()/255
  out=[]
  for p in pred:
   mae=(p-gt).abs().mean();lps=torch.cat([self.lp(p[i:i+2]*2-1,gt[i:i+2]*2-1).flatten() for i in range(0,len(gt),2)]).mean()
   out.append(dict(mae=float(mae),lpips=float(lps),reward=-float(mae+lps)))
  return out

def mix(m,s):
 assert m.shape==s.shape and torch.isfinite(m).all() and torch.isfinite(s).all()
 return .75*(m-m.mean())/m.std().clamp_min(1e-6)+.25*(s-s.mean())/s.std().clamp_min(1e-6)
def as_video(pred):return np.round(pred.permute(0,2,3,1).detach().cpu().clamp(0,1).numpy()*255).astype('uint8')
