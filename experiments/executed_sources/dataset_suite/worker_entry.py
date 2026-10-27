import os,sys,runpy
from pathlib import Path
v=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(v)==8 and len(set(v))==8
os.environ['CUDA_VISIBLE_DEVICES']=v[int(os.environ['LOCAL_RANK'])]
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
mode=sys.argv[1];assert mode in ['prepare','train','evaluate']
sys.argv=[str(Path(__file__).with_name(mode+'.py'))]+sys.argv[2:]
runpy.run_path(sys.argv[0],run_name='__main__')
