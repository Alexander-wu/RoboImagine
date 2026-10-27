"""Assign one GPU to each rank inside the Ray-reserved four-GPU allocation."""
import os,sys,runpy
from pathlib import Path
devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
mode=sys.argv[1]; assert mode in ['prepare','train','evaluate']
assert len(devices)==int(os.environ['WORLD_SIZE']) and len(set(devices))==len(devices)
assert len(devices)==4
os.environ['CUDA_VISIBLE_DEVICES']=devices[int(os.environ['LOCAL_RANK'])]
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
sys.argv=[str(Path(__file__).with_name(mode+'.py'))]+sys.argv[2:]
runpy.run_path(sys.argv[0],run_name='__main__')
