"""CPU audit against actual inherited functions, without loading a model."""
import ast,json,random
from pathlib import Path
import torch

root=Path(__file__).resolve().parents[2]
training=root/'rt1_mjev_20261011/reference/src/futureworlds/training.py'
memory=root/'roboalign_r2_text_20261010/reference/src/futureworlds/memory.py'
ns={'torch':torch}
for path,names in [(memory,{'retained','make_prefix'}),(training,{'sft_example'})]:
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
context=torch.arange(1280).reshape(1,1,1280)%4375
dynamics=torch.stack([torch.full((80,),i+20) for i in range(12)])
actions=torch.stack([torch.full((13,),8750+i) for i in range(12)])
proof=[]
for target in range(2,12):
    b=ns['sft_example'](context,dynamics,actions,target,anchor=1)
    kept=[1]+list(range(max(2,target-6),target))
    expected=torch.cat([context.reshape(1,-1)+4375]+
                       [v for i in kept for v in [dynamics[i:i+1],actions[i:i+1]]]+[dynamics[target:target+1]],1)
    assert torch.equal(b['input_ids'],expected)
    assert int((b['labels']!=-100).sum())==80
    assert torch.equal(b['labels'][:,-80:],dynamics[target:target+1])
    assert (b['labels'][:,:-80]==-100).all()
    assert (b['labels'][b['input_ids']>=4375]==-100).all()
    assert torch.equal(b['position_ids'],torch.arange(expected.shape[1])[None])
    proof.append(dict(target=target,kept=kept,tokens=expected.numel(),supervised=80))
rows=list(range(5200)); random.Random(20261011).shuffle(rows)
shards=[[rows[(step-1)*8+rank+micro*4] for step in range(1,501) for micro in range(2)] for rank in range(4)]
assert len(set(sum(shards,[])))==4000
assert [shards[rank][step*2+micro] for step in range(500) for micro in range(2) for rank in range(4)]==rows[:4000]
result=dict(state='PASS',teacher_forced_layout=proof,unique_windows=4000,
            causal_action_pairing=True,labels_only_dynamics=True,seeded_disjoint_rank_schedule=True,
            world_size=4,gradient_accumulation=2,global_batch=8,
            limitation='CPU layout audit only; GPU smoke must verify update, replica agreement, and checkpoint reload')
out=Path(__file__).with_name('local_checks.json'); out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
