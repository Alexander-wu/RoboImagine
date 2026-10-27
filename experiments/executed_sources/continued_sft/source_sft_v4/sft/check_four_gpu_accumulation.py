"""Actual four-rank CPU DDP vs serial global-batch-8 gradient/update check."""
import contextlib, json, tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def model():
    return torch.nn.Sequential(torch.nn.Linear(5,9),torch.nn.Tanh(),torch.nn.Linear(9,3)).double()

def run(rank,url,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=url,rank=rank,world_size=4)
    try:
        torch.manual_seed(1234);actor=model();reference=model();reference.load_state_dict(actor.state_dict())
        torch.manual_seed(20261011);x=torch.randn(8,10,5,dtype=torch.float64);y=torch.randint(0,3,(8,10))
        opt=torch.optim.AdamW(actor.parameters(),lr=1e-6,weight_decay=0.0)
        ropt=torch.optim.AdamW(reference.parameters(),lr=1e-6,weight_decay=0.0)
        ce=torch.nn.CrossEntropyLoss()
        ce(reference(x.reshape(80,5)),y.reshape(80)).backward()
        forward=DDP(actor,broadcast_buffers=False,gradient_as_bucket_view=True)
        for sample_micro in range(2):
            logical_rank=rank+sample_micro*4
            for frame_micro in range(10):
                sync=(sample_micro==1 and frame_micro==9)
                with contextlib.nullcontext() if sync else forward.no_sync():
                    loss=ce(forward(x[logical_rank,frame_micro:frame_micro+1]),y[logical_rank,frame_micro:frame_micro+1])/20
                    loss.backward()
        gradient_error=max(float((a.grad-b.grad).abs().max()) for a,b in zip(actor.parameters(),reference.parameters()))
        assert gradient_error<1e-12,gradient_error
        a_norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0));b_norm=float(torch.nn.utils.clip_grad_norm_(reference.parameters(),1.0))
        assert abs(a_norm-b_norm)<1e-12
        opt.step();ropt.step()
        update_error=max(float((a-b).detach().abs().max()) for a,b in zip(actor.parameters(),reference.parameters()))
        assert update_error<1e-12,update_error
        assert {int(v['step']) for v in opt.state.values()}=={1}
        proof=dict(rank=rank,gradient_max_abs_error=gradient_error,update_max_abs_error=update_error,
                   global_batch=8,world_size=4,accumulation=2,optimizer_updates=1)
        all_proofs=[None]*4;dist.all_gather_object(all_proofs,proof)
        if rank==0:Path(out).write_text(json.dumps(dict(state='PASS',proofs=all_proofs),indent=2)+'\n')
    finally:dist.destroy_process_group()

if __name__=='__main__':
    out=Path(__file__).with_name('four_gpu_accumulation_checks.json').resolve()
    with tempfile.TemporaryDirectory() as temp:
        mp.spawn(run,args=('file://'+str(Path(temp)/'rendezvous'),str(out)),nprocs=4,join=True)
    print(out.read_text())
