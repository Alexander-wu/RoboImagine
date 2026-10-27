"""CPU checks of the actual new helpers and sealed search/policy implementations.

Only function definitions are extracted from common.py to avoid its GPU/runtime
imports; the functions themselves are not reimplemented. No research artifacts
or shared caches are modified. Run with the suite's code/src, or --source-root.
"""
import argparse,ast,contextlib,copy,hashlib,json,os,sys,tempfile
from pathlib import Path
os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from safetensors.torch import save_file,load_file

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,default=ROOT/'code/src')
args=parser.parse_args();sys.path.insert(0,str(args.source_root))
from futureworlds.memory import rollout,make_prefix,frame_logp
from futureworlds.search import rollout_group_beam
from futureworlds.ordinary_search import rollout_ordinary_beam
from futureworlds.text_conditioning import TextWorldModel
from futureworlds.objective import compute_grpo_outcome_advantage,compute_policy_loss,kl_penalty
torch.set_num_threads(2);torch.manual_seed(991)

def extracted(path,names,namespace):
 tree=ast.parse(Path(path).read_text())
 nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
 assert {node.name for node in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
ns=dict(torch=torch,np=np,json=json,copy=copy,hashlib=hashlib,Path=Path,
        rollout=rollout,make_prefix=make_prefix,rollout_group_beam=rollout_group_beam,
        rollout_ordinary_beam=rollout_ordinary_beam,sha=sha,_DONORS=None,
        _CANDIDATE_SEQUENCE_KEYS={'model_log_scores','search_scores','frame_parent_indices','candidate_ids','candidate_scores','per_candidate'})
names=['digest','verify_assets','load_initial','_copy_tensors','_donor_lookup','text_donor','generator_instruction',
       'bind_generator_condition','_raw_cache','load_cache','cond','search_config','_select_candidate_fields','propose_candidates']
extracted(ROOT/'common.py',names,ns)
policy_ns=dict(torch=torch,np=np,contextlib=contextlib,frame_logp=frame_logp,
               compute_grpo_outcome_advantage=compute_grpo_outcome_advantage,
               compute_policy_loss=compute_policy_loss,kl_penalty=kl_penalty)
extracted(args.source_root/'futureworlds/training.py',['policy_backward'],policy_ns)
policy_backward=policy_ns['policy_backward']

cfg=LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                num_key_value_heads=2,vocab_size=9008,max_position_embeddings=1024,attention_dropout=0.)
def model():return TextWorldModel(LlamaForCausalLM(cfg),text_width=16,layers=(0,),inner=16).eval()
actor=model()
with torch.no_grad():actor.adapters['0'].out.weight.normal_(std=.01)
features=torch.randn(1,5,16);mask=torch.ones(1,5,dtype=torch.long)
actor.set_text(features,mask)
context=torch.randint(4375,8750,(1,8));anchor=torch.randint(0,4375,(1,80));actions=torch.randint(8750,9006,(10,13))
horizon=8;seed=171
results={}
for strategy in ['sample','ordinary','diverse']:
 ns['P']={'search':dict(strategy=strategy,active_paths=8,returned_candidates=4,groups=4,beams=2,
                         penalty=.1 if strategy=='diverse' else 0.,sample_selected_indices=[0,2,4,6])}
 calls=[]
 hook=actor.register_forward_pre_hook(lambda module,inputs,kw:calls.append(int(kw['input_ids'].shape[0])),with_kwargs=True)
 frames,trace=ns['propose_candidates'](actor,context,anchor,actions,horizon,seed);hook.remove()
 assert len(calls)==horizon*80 and set(calls)=={8}
 assert trace['actor_token_logit_rows']==sum(calls)==8*horizon*80
 raw_args=(actor,context,anchor,actions,horizon)
 if strategy=='sample':
  raw,_=rollout(*raw_args,group=8,generator=torch.Generator().manual_seed(seed))
  expected=[{k:(v[[0,2,4,6]] if torch.is_tensor(v) else v) for k,v in f.items()} for f in raw]
  assert trace['selected_generation_indices']==[0,2,4,6] and not trace['reward_based_selection']
 elif strategy=='ordinary':expected,_=rollout_ordinary_beam(*raw_args,penalty=0,groups=4,beams=2)
 else:expected,_=rollout_group_beam(*raw_args,penalty=.1,groups=4,beams=2)
 for actual,want in zip(frames,expected):
  assert actual.keys()==want.keys()
  for key,value in want.items():
   assert torch.equal(actual[key],value) if torch.is_tensor(value) else actual[key]==value
 # Target nine evicts state two from the rolling history; lineage must survive it.
 history={1:anchor.expand(4,-1),**{f['target']:f['response'] for f in frames}}
 for f in frames:
  rebuilt,kept=make_prefix(context.expand(4,-1),history,actions,f['target'])
  assert torch.equal(rebuilt,f['prefix'])
  saved=f.get('model_logp',f.get('sample_logp'))
  assert float((frame_logp(actor,f['prefix'],f['response'])-saved).detach().abs().max())<.002
 assert kept==[1,3,4,5,6,7,8]
 trained=model();trained.load_state_dict(actor.state_dict());trained.set_text(features,mask)
 reference=model();reference.load_state_dict(actor.state_dict());reference.set_text(features,mask);reference.requires_grad_(False)
 before=ns['digest'](trained);ref_before=ns['digest'](reference)
 opt=torch.optim.AdamW(trained.parameters(),lr=1e-4)
 loss=policy_backward(trained,trained,reference,frames,torch.tensor([-.1,-.2,-.4,-.8]),
                      dict(text_features=features,text_mask=mask),horizon,.2,.001)
 grad=float(torch.nn.utils.clip_grad_norm_(trained.parameters(),1,error_if_nonfinite=True));assert grad>0
 opt.step();assert ns['digest'](trained)!=before and ns['digest'](reference)==ref_before
 results[strategy]=dict(exact_baseline_or_fixed_slice_parity=True,long_history=True,logit_rows=sum(calls),loss=loss,gradient=grad)
 print('SEARCH_CHECK_PASS',strategy,flush=True)

# Candidate axes are sliced; the eight frame records and retained state IDs are not.
synthetic=[dict(prefix=torch.arange(24).reshape(8,3),retained=list(range(8)),
                model_log_scores=list(range(8)),per_candidate=[torch.arange(8) for _ in range(8)],
                nested={'response':torch.ones(8,80)}) for _ in range(8)]
selected=ns['_select_candidate_fields'](synthetic,[0,2,4,6],8)
assert len(selected)==8 and selected[0]['retained']==list(range(8))
assert selected[0]['model_log_scores']==[0,2,4,6] and selected[0]['prefix'].shape==(4,3)
assert len(selected[0]['per_candidate'])==4 and selected[0]['per_candidate'][0].shape==(8,)
selected[0]['prefix'].zero_();assert synthetic[0]['prefix'][1,0]==3

with tempfile.TemporaryDirectory(prefix='r2-search-logic-') as temp:
 root=Path(temp);selection={};mapping={};cache={}
 for split in ['train','smoke','dev']:
  rows=[]
  for i in range(2):
   key=f'{split}_{i}';row=dict(id=key,cache_id=key,text=f'{split} instruction {i}',path=key,sha256='inputhash',cache_sha256='cachehash')
   rows.append(row);mapping[key]=f'{split}_{1-i}'
   cache[key]=dict(context=torch.full((1,8),100+i),actions=torch.full((10,13),i),
                   features=torch.full((1,5,16),float(i)),mask=torch.ones(1,5,dtype=torch.long))
  selection[split]=rows
 (root/'selection.json').write_text(json.dumps(selection));donor_path=root/'text_donors.json';donor_path.write_text(json.dumps({'mapping':mapping}))
 ns.update(E=root,_DONORS=None,P={'text_mode':'shuffled','text_donor_map_file':str(donor_path),'text_donor_map_sha256':sha(donor_path)})
 real_raw=ns['_raw_cache'];ns['_raw_cache']=lambda row:ns['_copy_tensors'](cache[row['cache_id']])
 row=selection['train'][0];conditioned=ns['load_cache'](row)
 assert torch.equal(conditioned['features'],cache['train_1']['features'])
 assert torch.equal(conditioned['context'],cache['train_0']['context'])
 assert ns['generator_instruction'](row)==selection['train'][1]['text']
 ns['P']['text_mode']='no_text';conditioned=ns['load_cache'](row);assert not conditioned['mask'].any() and cache['train_0']['mask'].all()
 ns['P']['text_mode']='full';conditioned=ns['load_cache'](row);conditioned['features'].add_(3);assert cache['train_0']['features'].sum()==0
 ns['_raw_cache']=real_raw

 # Exercise strict core+adapter loading and the real condition wrapper on CPU.
 cp=root/'checkpoint';cp.mkdir();saved=model();saved.layers=[3,7,11]
 save_file({k:v.contiguous() for k,v in saved.core.state_dict().items()},str(cp/'model.safetensors'))
 save_file({k:v.contiguous() for k,v in saved.adapters.state_dict().items()},str(cp/'text_adapters.safetensors'))
 class TinyPipe:
  def __init__(self,*a,**kw):
   self.model=model();self.model.layers=[3,7,11];self.settings={'text_conditioning':{'enabled':True}};self.action_dim=13
   self.cached=dict(text_features=features.clone(),text_mask=mask.clone())
  def condition(self,text):self.model.set_text(self.cached['text_features'],self.cached['text_mask']);return self.cached
 ns.update(D=root,WorldPipeline=TinyPipe,load_file=lambda path,device:load_file(path,device='cpu'))
 ns['P'].update(dataset='rt1',action_dim=13,initial_checkpoint=str(cp),expected_initial_digest=ns['digest'](saved),
                verified_files={str(p):sha(p) for p in cp.iterdir()})
 ns['P']['text_mode']='no_text';pipe=ns['load_initial']();c=pipe.condition('correct instruction')
 assert not c['text_mask'].any() and pipe.cached['text_mask'].all()
 assert ns['digest'](pipe.model)==ns['digest'](saved)
 ns['P']['text_mode']='shuffled';pipe=ns['load_initial']();used=ns['bind_generator_condition'](pipe,row)
 assert used==selection['train'][1]['text'];pipe.condition(row['text'])

# All-zero masks suppress language exactly and keep adapters in the gradient graph.
zero=torch.zeros_like(mask);actor.set_text(features,zero);input_ids=torch.randint(0,9008,(1,12))
out1=actor(input_ids=input_ids,use_cache=False).logits
actor.set_text(features+99,zero);out2=actor(input_ids=input_ids,use_cache=False).logits
assert torch.equal(out1,out2)
actor.zero_grad(set_to_none=True);out2.square().mean().backward()
adapter_grads=[p.grad for p in actor.adapters.parameters()]
assert all(g is not None and torch.isfinite(g).all() and not g.any() for g in adapter_grads)
results['text_and_loader']=dict(cache_immutable=True,split_local_donors=True,strict_core_and_adapters=True,
                                zero_mask_text_invariance=True,zero_mask_adapter_gradients_present=True)
print('SEARCH_LOGIC_CHECKS='+json.dumps(results,allow_nan=False),flush=True)
