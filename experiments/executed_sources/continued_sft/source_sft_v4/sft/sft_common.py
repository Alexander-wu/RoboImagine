"""Independent, read-only reuse of the sealed R2 RT-1 runtime and cohorts."""
import os
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import sys, json, hashlib, random, datetime
from pathlib import Path

D = Path('/opt/roboimagine-workspace')
OLD = D/'experiments/rt1_mjev_20261011/sft_mjev500_8gpu_v1'
BASELINE = D/'experiments/rt1_mjev_20261011/sft_reference_v1/complete.json'
S = Path(__file__).resolve().parent
E = Path(os.environ.get('R2_SFT_OUTPUT', str(S/'rt1_sft500_v1')))
CODE = OLD/'code/src'
sys.path.insert(0, str(CODE))
CODEC = D/'experiments/infinitystar_robot_codec_20261008'
sys.path.insert(0, str(CODEC/'deps'))
sys.path.insert(0, str(D/'experiments/roboalign_r2_text_20261010/reward_compare_v1/eval_deps'))
os.environ['TORCH_HOME'] = str(CODEC/'torch_cache')
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from futureworlds.pipeline import WorldPipeline
from futureworlds.training import sft_example

SEED = 20261011
WORLD_SIZE = 4
GLOBAL_BATCH = 8
ACCUMULATION = GLOBAL_BATCH // WORLD_SIZE
EXPECTED_INITIAL = 'e6c3a7b5db0d677f55ece747ddec6edf7f60b5b901a95b14a8d575c8c5a27865'

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def setup(seed=SEED):
    torch.set_num_threads(2)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

def digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def protocol():
    return dict(version='rt1_sft_continuation500_v2_4gpu', dataset='rt1', seed=SEED,
                initializer='text-enabled RT-1 SFT50000, prior RL=0',
                initial_digest=EXPECTED_INITIAL, old_root=str(OLD), baseline=str(BASELINE),
                updates=500, world_size=WORLD_SIZE, global_batch=GLOBAL_BATCH,
                gradient_accumulation=ACCUMULATION,
                schedule='same seeded shuffle of the existing 5200 windows; first 4000',
                targets=list(range(2, 12)), memory='fixed initial context + dynamic anchor1 + recent6',
                objective='teacher-forced next-token CE on 80 dynamics tokens per frame; mean of 10 frames',
                optimizer=dict(type='AdamW', lr=1e-6, weight_decay=0.0, grad_clip=1.0, fresh=True),
                precision='FP32 weights and computation, TF32 disabled; matches RL update precision',
                text='existing trained cross-attention; fixed valid instruction; no text dropout',
                frozen='visual tokenizer and T5; train core and existing text adapters',
                checkpoints=[100, 250, 500], selection='final step500; no best-dev checkpoint selection',
                evaluation=dict(cases=128, horizon=32, beam_width=4, memory='full',
                                primary='mean predicted frames1..32, then all128 trajectories',
                                seed='20261011+10000+index', no_future_images=True),
                cost='matched optimizer updates and trajectory windows, not matched GPU time; '
                     'preparation, training and evaluation durations reported separately',
                scope='one synchronized 4-GPU training run with two examples per rank per update; no independent training-seed claim')

def verified_preparation(witnesses):
    """Admit immutable v3 caches using a narrowly checked hardware-only migration."""
    prepared=json.loads((E/'prepared.json').read_text())
    old_protocol=json.loads((E/'protocol.json').read_text())
    assert prepared['state']=='READY' and prepared['witnesses']==witnesses
    assert prepared['initial_digest']==EXPECTED_INITIAL
    assert prepared['protocol_sha256']==sha(E/'protocol.json')
    assert prepared['schedule_sha256']==sha(E/'schedule.json')
    migration=False
    if prepared['source']==own_seal():
        assert old_protocol==protocol()
    else:
        legacy=S.parent.parent/'source_sft_v3/sft'
        assert legacy.is_dir(), 'Only the verified immutable v3 preparation may migrate'
        assert prepared['source']=={p.name:sha(p) for p in legacy.glob('*.py')}
        assert old_protocol['version']=='rt1_sft_continuation500_v1'
        assert old_protocol['world_size']==8 and old_protocol['global_batch']==GLOBAL_BATCH
        allowed={'version','world_size','gradient_accumulation','scope'}
        assert {k:v for k,v in old_protocol.items() if k not in allowed}=={
            k:v for k,v in protocol().items() if k not in allowed}, 'Only hardware/accumulation changes are permitted'
        migration=True
    admission=dict(state='READY',hardware_only_migration=migration,
                   original_prepared_sha256=sha(E/'prepared.json'),
                   original_protocol_sha256=sha(E/'protocol.json'),
                   original_preparation_source=prepared['source'],
                   current_source=own_seal(),current_protocol=protocol(),
                   schedule_sha256=sha(E/'schedule.json'),witnesses=witnesses,
                   cache_policy='Original prepared.json, protocol.json, cache files and receipts remain unchanged')
    return prepared,admission

def verified_inputs():
    original = json.loads((OLD/'protocol.json').read_text())
    seal = json.loads((OLD/'source_manifest.json').read_text())
    for name, expected in seal.items():
        assert sha(OLD/name) == expected, ('old source changed', name)
    for name, info in original['assets'].items():
        if not name.startswith('rt1/models/main/'):
            assert sha(D/'bundle'/name) == info['sha256'], name
    prepared = json.loads((OLD/'prepared.json').read_text())
    assert sha(OLD/'selection.json') == prepared['selection_sha256']
    assert prepared['initial_digest'] == EXPECTED_INITIAL
    selection = json.loads((OLD/'selection.json').read_text())
    base = json.loads(BASELINE.read_text())
    assert base['initial_digest'] == EXPECTED_INITIAL
    assert len(selection['train']) == 5200 and len(selection['smoke']) == 8 and len(selection['dev']) == 128
    assert [x['id'] for x in selection['dev']] == [x['id'] for x in base['evaluation']]
    assert all(x['sha256'] == y['prepared_input_sha256'] for x,y in zip(selection['dev'], base['evaluation']))
    rows = list(selection['train']); random.Random(SEED).shuffle(rows)
    assert len({x['id'] for x in rows}) == 5200
    assert not {x['id'] for x in rows} & {x['id'] for x in selection['dev']}
    # If the old run schedule is present, audit exact order rather than assume it.
    old_schedule = OLD/'run/schedule.json'
    assert old_schedule.exists(), 'Need the actual completed RL schedule'
    assert [x['id'] for x in rows] == json.loads(old_schedule.read_text())
    old_complete = OLD/'run/complete.json'
    run = json.loads(old_complete.read_text())
    assert run['state']=='SUCCEEDED' and run['updates']==500 and run['global_batches']==500
    assert run['skipped']==0 and run['groups']==4000 and run['initial_digest']==EXPECTED_INITIAL
    witnesses = dict(selection_sha256=sha(OLD/'selection.json'),
                     prepared_sha256=sha(OLD/'prepared.json'), baseline_sha256=sha(BASELINE),
                     old_complete_sha256=sha(old_complete), old_global_batches=500,
                     old_source_manifest_sha256=sha(OLD/'source_manifest.json'),
                     runtime_source={str(p.relative_to(CODE)): sha(p) for p in CODE.rglob('*.py')})
    return selection, rows, witnesses

def pixels(row):
    assert sha(row['path']) == row['sha256']
    with np.load(row['path'], allow_pickle=False) as z:
        images=z['image'].copy(); actions=z['action'].astype('float32').copy(); text=str(z['instruction'])
    assert images.shape == (row['clip_length'],256,320,3) and images.dtype == np.uint8
    assert text == row['text'] and actions.shape == (len(images),13) and np.isfinite(actions).all()
    return images, actions, text

def load_initial():
    p = WorldPipeline(D/'bundle', 'rt1', 'sft', 'cuda:0', str(D/'text_model'), verify=False)
    assert p.settings['text_conditioning']['enabled'] and p.model.layers == [3,7,11]
    assert p.settings['variants']['sft']['updates'] == 50000
    assert p.settings['variants']['sft']['method'] == 'SFT'
    assert digest(p.model) == EXPECTED_INITIAL
    p.model.eval(); return p

def load_cache(row):
    path=E/'gt_cache'/f"{row['cache_id']}.pt"
    receipt=json.loads(path.with_suffix('.json').read_text())
    assert receipt['input_sha256']==row['sha256'] and receipt['cache_sha256']==sha(path)
    return torch.load(path, map_location='cuda:0', weights_only=True)

def cond(cache):
    return dict(text_features=cache['features'], text_mask=cache['mask'])

def init_ddp(allow_prepare=False):
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    assert world==WORLD_SIZE
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=torch.device('cuda:0'))
    return rank, world

def gather(value):
    output=[None]*dist.get_world_size(); dist.all_gather_object(output,value); return output

def agree(model):
    value=digest(model); values=gather(value); assert len(set(values))==1, values; return value

def own_seal():
    return {p.name:sha(p) for p in S.glob('*.py')}
