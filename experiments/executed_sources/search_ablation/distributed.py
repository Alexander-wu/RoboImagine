from common import *
import datetime,contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
RANK=int(os.environ['RANK']);WORLD=int(os.environ['WORLD_SIZE']);assert WORLD==8
torch.cuda.set_device(0)
dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=12),device_id=torch.device('cuda:0'))
def gather(x):
 out=[None]*WORLD;dist.all_gather_object(out,x);return out
def agree(model):
 value=digest(model);values=gather(value);assert len(set(values))==1,values;return value
def verify_source():
 s=json.loads((ROOT/'source_manifest.json').read_text());assert all(sha(ROOT/n)==h for n,h in s.items());return s
