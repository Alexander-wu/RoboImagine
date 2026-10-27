"""Cache recorded future tokens in a NEW directory, with exact anchor parity."""
from sft_common import *
import time, traceback, uuid

rank,world=init_ddp(allow_prepare=True); setup(); started=time.monotonic()
selection,rows,witnesses=verified_inputs()
E.mkdir(parents=True,exist_ok=True)
if (E/'prepared.json').exists():
    prior,admission=verified_preparation(witnesses)
    for i,row in enumerate(rows[:4000]+selection['smoke']):
        if i%world!=rank: continue
        path=E/'gt_cache'/f"{row['cache_id']}.pt"
        receipt=json.loads(path.with_suffix('.json').read_text())
        assert receipt['input_sha256']==row['sha256'] and receipt['cache_sha256']==sha(path)
        assert receipt['context_anchor_actions_text_exact'] and receipt['teacher_future_frames']==10
    dist.barrier()
    if rank==0:
        path=E/'four_gpu_preparation_admission.json'
        if path.exists():assert json.loads(path.read_text())==admission
        else:write(path,admission)
    dist.barrier(); print('SFT_PREPARATION_ALREADY_READY',rank,flush=True)
    dist.destroy_process_group(); sys.exit(0)
if rank==0:
    p=E/'protocol.json'
    if p.exists(): assert json.loads(p.read_text())==protocol(), 'Protocol changed'
    else: write(p,protocol())
    write(E/'schedule.json',[x['id'] for x in rows[:4000]])
    write(E/'input_witnesses.json',witnesses)
dist.barrier()
pipe=load_initial(); assert agree(pipe.model)==EXPECTED_INITIAL
items=rows[:4000]+selection['smoke']; receipts=[]
try:
    for index,row in enumerate(items):
        if index%world!=rank: continue
        p=E/'gt_cache'/f"{row['cache_id']}.pt"; p.parent.mkdir(exist_ok=True)
        rp=p.with_suffix('.json')
        if p.exists() and rp.exists():
            receipt=json.loads(rp.read_text())
            assert receipt['input_sha256']==row['sha256'] and receipt['cache_sha256']==sha(p)
        else:
            images,actions,text=pixels(row)
            old_path=OLD/'cache'/f"{row['cache_id']}.pt"
            assert sha(old_path)==row['cache_sha256']
            old=torch.load(old_path,map_location='cuda:0',weights_only=True)
            # RL caches its observed anchor through the native two-frame path.
            # Batch-8 future encoding can change a quantizer rounding boundary;
            # independently verify, then preserve that exact observed input.
            observed_ctx,observed_anchor=pipe.encode_observations(images[:2])
            assert torch.equal(observed_ctx,old['context'])
            assert torch.equal(observed_anchor,old['dynamics'][1:2])
            ctx,dyn,acts,_,co=pipe.encode_training_window(images,actions,text)
            assert dyn.shape==(12,80)
            future_before=dyn[2:].clone()
            batched_anchor_mismatches=int((dyn[1:2]!=observed_anchor).sum().item())
            dyn[1:2]=observed_anchor
            assert torch.equal(dyn[2:],future_before)
            # No altered observation encoder or action mapping is admitted.
            assert torch.equal(ctx,old['context'])
            assert torch.equal(dyn[1:2],old['dynamics'][1:2])
            assert torch.equal(acts,old['actions'])
            assert torch.equal(co['text_features'],old['features']) and torch.equal(co['text_mask'],old['mask'])
            value=dict(context=ctx.cpu(),dynamics=dyn.cpu(),actions=acts.cpu(),
                       features=co['text_features'].cpu(),mask=co['text_mask'].cpu())
            tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial'); torch.save(value,tmp); os.replace(tmp,p)
            receipt=dict(id=row['id'],input_sha256=row['sha256'],old_cache_sha256=row['cache_sha256'],
                         cache_sha256=sha(p),context_anchor_actions_text_exact=True,teacher_future_frames=10,
                         observed_anchor_path='native two-frame encoding, exactly matched to original RL cache',
                         batched_anchor_mismatches=batched_anchor_mismatches,
                         future_targets_unchanged=True)
            write(rp,receipt)
            reloaded=torch.load(p,map_location='cuda:0',weights_only=True)
            assert torch.equal(reloaded['dynamics'][1:2],old['dynamics'][1:2])
            assert torch.equal(reloaded['dynamics'][2:],future_before)
            del old,ctx,dyn,acts,co,value,observed_ctx,observed_anchor,future_before,reloaded
        receipts.append(receipt)
        if len(receipts)%20==0:
            write(E/f'prepare_rank{rank}.json',dict(state='PREPARING',cases=len(receipts),total=4008//world,
                                                  elapsed_seconds=time.monotonic()-started))
            print('SFT_PREPARE',rank,len(receipts),4008//world,flush=True)
        pipe.text_cache.clear()
    assert len(receipts)==4008//world and agree(pipe.model)==EXPECTED_INITIAL
    parts=gather(dict(rank=rank,receipts=receipts,seconds=time.monotonic()-started))
    if rank==0:
        combined=[v for part in parts for v in part['receipts']]
        assert len(combined)==4008 and len({v['id'] for v in combined})==4008
        write(E/'prepared.json',dict(state='READY',train_windows=4000,smoke_windows=8,
                                    initial_digest=EXPECTED_INITIAL,protocol_sha256=sha(E/'protocol.json'),
                                    schedule_sha256=sha(E/'schedule.json'),witnesses=witnesses,source=own_seal(),
                                    preparation_seconds=max(x['seconds'] for x in parts),
                                    preparation_world_size=world,
                                    observed_anchor='exact original RL two-frame input; future targets retain native block encoding',
                                    batched_anchor_mismatch_windows=sum(v.get('batched_anchor_mismatches',0)>0 for v in combined),
                                    receipts=combined))
    write(E/f'prepare_rank{rank}.json',dict(state='SUCCEEDED',cases=4008//world,total=4008//world))
    dist.barrier()
    _,admission=verified_preparation(witnesses)
    if rank==0:write(E/'four_gpu_preparation_admission.json',admission)
    dist.barrier()
except BaseException:
    write(E/f'prepare_ERROR_rank{rank}.json',dict(error=traceback.format_exc()))
    traceback.print_exc()
    # Let torchrun stop peer ranks promptly instead of hanging in NCCL teardown.
    os._exit(1)
finally:
    dist.destroy_process_group()
