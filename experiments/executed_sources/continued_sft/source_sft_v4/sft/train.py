"""Matched 500-update SFT continuation with immutable, resumable checkpoints."""
from sft_common import *
import argparse, contextlib, gc, time, traceback, uuid
from safetensors.torch import save_file, load_file

parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
args=parser.parse_args(); rank,world=init_ddp(); setup()
selection,rows,witnesses=verified_inputs()
prepared,admission=verified_preparation(witnesses)
assert json.loads((E/'four_gpu_preparation_admission.json').read_text())==admission
assert json.loads((E/'schedule.json').read_text())==[x['id'] for x in rows[:4000]]
if not args.smoke:
    gate=json.loads((E/'gate.json').read_text())
    assert gate['state']=='PASS' and gate['source']==own_seal()
O=E/('smoke4' if args.smoke else 'run'); O.mkdir(parents=True,exist_ok=True)
done=O/'training_complete.json'
if done.exists():
    result=json.loads(done.read_text()); assert result['state']=='SUCCEEDED'
    assert result['source']==own_seal() and result['protocol']==protocol()
    print('SFT_ALREADY_TRAINED',str(done),flush=True)
    dist.destroy_process_group(); sys.exit(0)

attempt=uuid.uuid4().hex[:12] if rank==0 else None
ids=[attempt]; dist.broadcast_object_list(ids,src=0); attempt=ids[0]
R=O/'attempts'/attempt/f'rank{rank}'; R.mkdir(parents=True,exist_ok=False)
pipe=load_initial(); actor=pipe.model; initial=agree(actor)
actor.train(); forward=DDP(actor,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
opt=torch.optim.AdamW(actor.parameters(),lr=1e-6,weight_decay=0.0)
step=0; previous_seconds=0.0; nonzero_action_grad=None; nonzero_adapter_grad=None
checkpoints=sorted(p for p in O.glob('checkpoint_*') if p.is_dir() and p.name[-6:].isdigit() and (p/'receipt.json').exists())
if checkpoints:
    cp=checkpoints[-1]; receipt=json.loads((cp/'receipt.json').read_text())
    assert receipt['source']==own_seal() and receipt['initial_digest']==initial
    assert all(sha(cp/name)==expected for name,expected in receipt['files'].items())
    actor.core.load_state_dict(load_file(str(cp/'model.safetensors'),device='cuda:0'),strict=True)
    actor.adapters.load_state_dict(load_file(str(cp/'text_adapters.safetensors'),device='cuda:0'),strict=True)
    assert agree(actor)==receipt['digest']
    state=torch.load(cp/'training_state.pt',map_location='cpu',weights_only=False)
    assert state['world_size']==world and state['global_batch']==GLOBAL_BATCH
    assert state['gradient_accumulation']==ACCUMULATION and state['protocol']==protocol()
    opt.load_state_dict(state['optimizer'])
    step=state['updates']; previous_seconds=state['training_seconds']
    if args.smoke:
        nonzero_action_grad=state['smoke_gradients']['action']
        nonzero_adapter_grad=state['smoke_gradients']['adapters']
    rng=state['rng'][rank]; assert rng['rank']==rank
    torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'])
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    print('SFT_RESUME',rank,step,str(cp),flush=True)
else:
    assert initial==EXPECTED_INITIAL

started=time.monotonic(); target_steps=1 if args.smoke else 500
torch.cuda.reset_peak_memory_stats()
def status(state,**kw):
    data=dict(state=state,rank=rank,attempt=attempt,updates=step,target_updates=target_steps,
              elapsed_seconds=previous_seconds+time.monotonic()-started,**kw)
    write(R/'status.json',data)
    if rank==0: write(O/'status.json',data)

def checkpoint():
    replica=agree(actor)
    states=gather(dict(rank=rank,torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                       python=random.getstate(),numpy=np.random.get_state()))
    elapsed=previous_seconds+time.monotonic()-started
    if rank==0:
        cp=O/f'checkpoint_{step:06d}'; assert not cp.exists(), ('No checkpoint overwrite',str(cp))
        tmp=cp.with_name(cp.name+'.'+attempt+'.partial'); tmp.mkdir(exist_ok=False)
        actor.core.save_pretrained(tmp)
        save_file({k:v.detach().cpu().contiguous() for k,v in actor.adapters.state_dict().items()},str(tmp/'text_adapters.safetensors'))
        torch.save(dict(updates=step,optimizer=opt.state_dict(),rng=states,world_size=world,global_batch=GLOBAL_BATCH,gradient_accumulation=ACCUMULATION,
                        protocol=protocol(),training_seconds=elapsed,
                        smoke_gradients=dict(action=nonzero_action_grad,adapters=nonzero_adapter_grad)),tmp/'training_state.pt')
        write(tmp/'receipt.json',dict(updates=step,digest=replica,initial_digest=initial,
                                      source=own_seal(),protocol=protocol(),attempt=attempt,
                                      files={p.name:sha(p) for p in tmp.iterdir() if p.is_file()}))
        os.replace(tmp,cp)
    dist.barrier(); return replica

try:
    status('TRAINING')
    for batch in range(step+1,target_steps+1):
        opt.zero_grad(set_to_none=True); tick=time.monotonic()
        total=0.0; nonzero_action_grad=None; nonzero_adapter_grad=None; batch_rows=[]
        for sample_micro in range(ACCUMULATION):
            logical_rank=rank+sample_micro*world
            row=selection['smoke'][logical_rank] if args.smoke else rows[(batch-1)*GLOBAL_BATCH+logical_rank]
            batch_rows.append(row)
            # Preserve the original per-example seed regardless of physical rank.
            setup(SEED+batch*GLOBAL_BATCH+logical_rank)
            c=load_cache(row); assert c['dynamics'].shape==(12,80)
            for frame_micro,target in enumerate(range(2,12)):
                b=sft_example(c['context'],c['dynamics'],c['actions'],target,anchor=1)
                assert int((b['labels']!=-100).sum())==80
                assert torch.equal(b['labels'][:,-80:],c['dynamics'][target:target+1])
                assert (b['labels'][b['input_ids']>=4375]==-100).all()
                synchronize=(sample_micro==ACCUMULATION-1 and frame_micro==9)
                with contextlib.nullcontext() if synchronize else forward.no_sync():
                    loss=forward(**b,**cond(c),use_cache=False).loss/(10*ACCUMULATION)
                    assert torch.isfinite(loss), ('Nonfinite SFT loss',batch,logical_rank,target)
                    loss.backward(); total+=float(loss.detach())
                del loss,b
            del c
        if args.smoke:
            nonzero_action_grad=float(actor.core.model.embed_tokens.weight.grad[8750:9006].norm())
            nonzero_adapter_grad={k:float(v.out.weight.grad.norm()) for k,v in actor.adapters.items()}
            assert nonzero_action_grad>0 and all(v>0 for v in nonzero_adapter_grad.values())
        grad=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0,error_if_nonfinite=True)); assert grad>0
        opt.step(); step+=1; torch.cuda.synchronize()
        measured=torch.tensor([total,time.monotonic()-tick],device='cuda:0',dtype=torch.float64)
        dist.all_reduce(measured); measured/=world
        record=dict(batch=batch,updates=step,rank=rank,ids=[r['id'] for r in batch_rows],
                    prepared_input_sha256=[r['sha256'] for r in batch_rows],global_batch=GLOBAL_BATCH,
                    gradient_accumulation=ACCUMULATION,
                    local_ce=total,mean_ce=float(measured[0]),mean_update_seconds=float(measured[1]),
                    gradient_norm=grad,targets=list(range(2,12)),supervised_tokens=800*ACCUMULATION)
        with (R/'train.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
        status('TRAINING',loss=float(measured[0]),gradient_norm=grad)
        if args.smoke or step in [100,250,500]: checkpoint()
        if step==1 or step%10==0: print('SFT_STEP',rank,step,float(measured[0]),float(measured[1]),flush=True)
    assert step==target_steps
    assert {int(v['step']) for v in opt.state.values()}=={target_steps}
    final=agree(actor); assert final!=initial
    training_seconds=previous_seconds+time.monotonic()-started
    train_peak=torch.cuda.max_memory_allocated()/2**30
    if args.smoke:
        row=selection['smoke'][rank]; images,actions,text=pixels(row)
        actor.eval(); probe=pipe.predict(images[:2].copy(),actions[:11].copy(),text,horizon=10,beam_width=4,memory='full')
        assert probe['prediction'].shape==(10,256,320,3) and np.isfinite(probe['prediction']).all()
        cp=O/'checkpoint_000001'
        actor.core.load_state_dict(load_file(str(cp/'model.safetensors'),device='cuda:0'),strict=True)
        actor.adapters.load_state_dict(load_file(str(cp/'text_adapters.safetensors'),device='cuda:0'),strict=True)
        assert digest(actor)==final
        own=dict(rank=rank,final_digest=final,action_gradient=nonzero_action_grad,
                 adapter_gradients=nonzero_adapter_grad,checkpoint_reload=True,inference=True)
        proofs=gather(own)
        if rank==0:
            write(E/'gate.json',dict(state='PASS',source=own_seal(),initial_digest=initial,
                                     final_digest=final,proofs=proofs,scope='4GPU x2-example accumulation smoke; formal run reloads original SFT'))
    all_cost=gather(dict(rank=rank,training_seconds=training_seconds,peak_allocated_GiB=train_peak))
    if rank==0:
        write(done,dict(state='SUCCEEDED',arm='sft_continuation',updates=step,world_size=world,global_batch=GLOBAL_BATCH,gradient_accumulation=ACCUMULATION,
                         initial_digest=initial,final_digest=final,protocol=protocol(),source=own_seal(),
                         train_seconds=max(x['training_seconds'] for x in all_cost),
                         train_gpu_hours=max(x['training_seconds'] for x in all_cost)*world/3600,
                         cost_caveat='resumed checkpoint time excludes lost work after last checkpoint; retain attempt logs',
                         peak_train_allocated_GiB=max(x['peak_allocated_GiB'] for x in all_cost),
                         checkpoint=str(O/f'checkpoint_{step:06d}')))
    status('SUCCEEDED'); dist.barrier()
except BaseException:
    write(R/'ERROR.json',dict(error=traceback.format_exc(),updates=step))
    status('FAILED'); traceback.print_exc(); os._exit(1)
finally:
    dist.destroy_process_group()
