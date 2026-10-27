
import os,sys,json,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent
os.environ.update(USE_TF='0',USE_FLAX='0');sys.path.insert(0,str(E/'code/src'))
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from futureworlds.text_conditioning import TextWorldModel
from futureworlds.memory import rollout
from futureworlds.search import rollout_group_beam
from futureworlds.ordinary_search import rollout_ordinary_beam
from futureworlds.training import policy_backward
torch.set_num_threads(2);torch.manual_seed(991)
cfg=LlamaConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,vocab_size=9008,max_position_embeddings=512,attention_dropout=0.)
def model():return TextWorldModel(LlamaForCausalLM(cfg),text_width=32,layers=(0,),inner=16).eval()
def digest(m):return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
base=model()
with torch.no_grad():base.adapters['0'].out.weight.normal_(std=.01)
results={}
for strategy in ['sample','ordinary','diverse']:
 actor=model();actor.load_state_dict(base.state_dict());reference=model();reference.load_state_dict(base.state_dict());reference.requires_grad_(False)
 ftr=torch.randn(1,5,32);mask=torch.ones(1,5,dtype=torch.long);co={'text_features':ftr,'text_mask':mask}
 actor.set_text(ftr,mask);reference.set_text(ftr,mask)
 context=torch.randint(4375,8750,(1,8));anchor=torch.randint(0,4375,(1,80));actions=torch.randint(8750,9006,(4,13))
 args=(actor,context,anchor,actions,2)
 if strategy=='sample':frames,trace=rollout(*args,group=4,generator=torch.Generator().manual_seed(13))
 elif strategy=='ordinary':frames,trace=rollout_ordinary_beam(*args,penalty=0,groups=4,beams=2)
 else:frames,trace=rollout_group_beam(*args,penalty=.1,groups=4,beams=2)
 old=digest(actor);ref=digest(reference);opt=torch.optim.AdamW(actor.parameters(),lr=1e-4)
 loss=policy_backward(actor,actor,reference,frames,torch.tensor([-.1,-.2,-.4,-.8]),co,2,.2,.01)
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.adapters.parameters())
 grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1,error_if_nonfinite=True));opt.step()
 assert digest(reference)==ref and digest(actor)!=old
 results[strategy]={'finite_gradient':True,'history_consistency_checks_pass':True,'actor_changed':True,'reference_unchanged':True,'gradient_norm':grad,'loss':loss}
 print('CPU_RL_STRATEGY_PASS',strategy,flush=True)
(E/'cpu_rl_check.json').write_text(json.dumps(results,indent=2))
print('CPU_RL_CHECK='+json.dumps(results),flush=True)
