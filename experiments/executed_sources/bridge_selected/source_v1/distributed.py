"""Four physical ranks, with two virtual old-protocol examples per rank."""
from common import *
import contextlib
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

RANK = int(os.environ['RANK'])
WORLD = int(os.environ['WORLD_SIZE'])
assert WORLD == 4, 'This experiment is capped at four GPUs'
torch.cuda.set_device(0)
dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=torch.device('cuda:0'))

def gather(value):
    out = [None] * WORLD
    dist.all_gather_object(out, value)
    return out

def agree(model):
    value = digest(model)
    values = gather(value)
    assert len(set(values)) == 1, values
    return value

def verify_source():
    source = json.loads((ROOT / 'source_manifest.json').read_text())
    entries = source.get('files', source)
    assert entries and all(sha(ROOT / name) == (value['sha256'] if isinstance(value, dict) else value) for name, value in entries.items())
    return source
