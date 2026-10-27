"""Map exactly one of the four allocated devices to each torchrun worker."""
import os
import runpy
import sys
from pathlib import Path

visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(visible) == len(set(visible)) == 4
local_rank = int(os.environ['LOCAL_RANK'])
assert 0 <= local_rank < 4
os.environ['CUDA_VISIBLE_DEVICES'] = visible[local_rank]
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
mode = sys.argv[1]
assert mode in ['train_mcts']
sys.argv = [str(Path(__file__).with_name(mode + '.py'))] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
