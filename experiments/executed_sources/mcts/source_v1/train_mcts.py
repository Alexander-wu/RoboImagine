"""RT-1 MCTS post-training; four GPUs x two trajectories = global batch eight.

The actor remains immutable while all eight proposal/reward groups are built.
The two local losses each have weight 1/2; DDP's rank averaging supplies 1/4.
Resume always writes a new output directory and pins its complete checkpoint.
"""
from distributed import *
import argparse
import gc
import math
import signal
import time
import traceback
from judge_client import Judge
from safetensors.torch import load_file, save_file
from futureworlds.evaluation import Metrics

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--resume', type=Path)
args = parser.parse_args()
assert not (args.smoke and args.resume)
ARM = P['arm']
assert ARM == 'hybrid'
assert P.get('text_mode', 'full') == 'full'
PROTOCOL_SHA256 = sha(E / 'protocol.json')
EXPECTED_INITIAL = P['expected_initial_digest']
assert isinstance(EXPECTED_INITIAL, str) and len(EXPECTED_INITIAL) == 64
assert P['optimizer']['lr'] == 1e-6 and P['clip'] == .2 and P['kl'] == .001
O = Path(os.environ['R2_OUTPUT_ROOT']).resolve()
assert O != E.resolve() and O != ROOT.resolve()
R = O / f'rank{RANK}'


def durable_jsonl(path, row):
    with path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def stopped(signum, frame):
    raise RuntimeError(f'Worker stopped by signal {signum}; output is preserved')


signal.signal(signal.SIGTERM, stopped)
signal.signal(signal.SIGINT, stopped)
setup(20261011)
source = verify_source()
prepared = json.loads((E / 'prepared.json').read_text())
assert sha(E / 'selection.json') == prepared['selection_sha256']
sel = json.loads((E / 'selection.json').read_text())
rows = list(sel['train'])
random.Random(20261011).shuffle(rows)
assert len(rows) == 5200 and len(sel['smoke']) == 8 and len(sel['dev']) == 128
if not args.smoke:
    gate = json.loads((E / 'gate.json').read_text())
    assert gate['state'] == 'PASS' and gate['initial_digest'] == EXPECTED_INITIAL
    assert gate['world'] == 4 and gate['global_batch'] == 8
    assert gate['protocol_sha256'] == PROTOCOL_SHA256
if RANK == 0:
    O.mkdir(parents=True, exist_ok=False)
dist.barrier()
R.mkdir(exist_ok=False)

pipe = load_initial()
actor = pipe.model
initial = agree(actor)
assert initial == EXPECTED_INITIAL == prepared['initial_digest']
refpipe = load_initial()
reference = refpipe.model.eval().requires_grad_(False)
assert digest(reference) == initial
updates = 0
last_batch = 0
first_batch = 1
local_ties = 0
global_skipped = 0
inherited_seconds = 0.0
inherited_peak = 0.0
resume_info = None
resume_state = None

if args.resume:
    cp = args.resume.resolve()
    assert cp.is_dir() and not cp.name.endswith('.partial')
    receipt = json.loads((cp / 'receipt.json').read_text())
    assert receipt['initial_digest'] == initial and receipt['protocol'] == P
    assert receipt['protocol_sha256'] == PROTOCOL_SHA256
    assert {'model.safetensors', 'text_adapters.safetensors', 'training_state.pt'}.issubset(receipt['files'])
    for name, expected in receipt['files'].items():
        assert sha(cp / name) == expected, (cp, name)
    resume_state = torch.load(cp / 'training_state.pt', map_location='cpu', weights_only=False)
    assert resume_state['world_size'] == 4 and resume_state['global_batch'] == 8
    assert resume_state['protocol'] == P and resume_state['protocol_sha256'] == PROTOCOL_SHA256
    updates = int(resume_state['updates'])
    last_batch = int(resume_state['global_batches'])
    assert 0 < updates <= 500 and updates <= last_batch <= P['max_global_batches']
    assert updates == receipt['updates'] and last_batch == receipt['global_batches']
    first_batch = last_batch + 1
    actor.core.load_state_dict(load_file(str(cp / 'model.safetensors'), device='cuda:0'), strict=True)
    actor.adapters.load_state_dict(load_file(str(cp / 'text_adapters.safetensors'), device='cuda:0'), strict=True)
    assert agree(actor) == receipt['digest']
    saved_ranks = {v['rank']: v for v in resume_state['ranks']}
    assert set(saved_ranks) == set(range(4))
    saved = saved_ranks[RANK]
    local_ties = int(saved['local_ties'])
    global_skipped = int(saved['global_skipped'])
    assert last_batch - global_skipped == updates
    inherited_seconds = float(saved['elapsed_train_seconds'])
    inherited_peak = float(saved['peak_train_allocated_GiB'])
    resume_info = dict(checkpoint=str(cp), receipt_sha256=sha(cp / 'receipt.json'),
                       source_output=str(cp.parent), resumed_from_update=updates,
                       resumed_from_global_batch=last_batch, checkpoint_digest=receipt['digest'])

forward = DDP(actor, device_ids=[0], broadcast_buffers=False, gradient_as_bucket_view=True)
opt = torch.optim.AdamW(actor.parameters(), lr=P['optimizer']['lr'], weight_decay=0)
if resume_state is not None:
    opt.load_state_dict(resume_state['optimizer'])
    # load_state_dict casts parameter-shaped optimizer tensors to CUDA; explicitly
    # assert that moments follow their parameters instead of silently retaining CPU.
    for parameter, state in opt.state.items():
        for name in ('exp_avg', 'exp_avg_sq'):
            assert name in state and state[name].device == parameter.device
    saved = {v['rank']: v for v in resume_state['ranks']}[RANK]
    torch.set_rng_state(saved['torch'])
    torch.cuda.set_rng_state(saved['cuda'])
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    del resume_state
judge = None
start = time.monotonic()
training_start = None
if RANK == 0:
    write(O / 'initial.json', dict(digest=initial, actor_at_start_digest=digest(actor), protocol=P,
          protocol_sha256=PROTOCOL_SHA256, source=source, seed=20261011, world=4, global_batch=8,
          gradient_accumulation=2, resume=resume_info, torch=torch.__version__,
          transformers=__import__('transformers').__version__))
    write(O / 'schedule.json', [row['id'] for row in rows])


def status(**data):
    write(R / 'status.json', dict(rank=RANK, **data))
    if RANK == 0:
        write(O / 'status.json', dict(world=4, global_batch=8, **data))


def elapsed_train():
    return inherited_seconds + (time.monotonic() - training_start if training_start is not None else 0.0)


def checkpoint(step, batch):
    current = agree(actor)
    states = gather(dict(rank=RANK, torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
        python=random.getstate(), numpy=np.random.get_state(), local_ties=local_ties,
        global_skipped=global_skipped, elapsed_train_seconds=elapsed_train(),
        peak_train_allocated_GiB=max(inherited_peak, float(torch.cuda.max_memory_allocated() / 2**30))))
    if RANK == 0:
        cp = O / f'checkpoint_{step:06d}'
        tmp = cp.with_name(cp.name + '.partial')
        tmp.mkdir(exist_ok=False)
        actor.core.save_pretrained(tmp)
        save_file({k: v.detach().cpu().contiguous() for k, v in actor.adapters.state_dict().items()},
                  str(tmp / 'text_adapters.safetensors'))
        torch.save(dict(updates=step, global_batches=batch, optimizer=opt.state_dict(), ranks=states,
                        world_size=4, global_batch=8, protocol=P, protocol_sha256=PROTOCOL_SHA256,
                        resume=resume_info), tmp / 'training_state.pt')
        write(tmp / 'receipt.json', dict(updates=step, global_batches=batch, digest=current,
              initial_digest=initial, protocol=P, protocol_sha256=PROTOCOL_SHA256, resume=resume_info,
              files={f.name: sha(f) for f in tmp.iterdir() if f.is_file()}))
        tmp.replace(cp)
    dist.barrier()
    return current


def make_group(batch, micro):
    virtual_rank = RANK + micro * WORLD
    row = sel['smoke'][virtual_rank] if args.smoke else rows[(batch - 1) * 8 + virtual_rank]
    seed = 20261011 + batch * 8 + virtual_rank
    setup(seed)
    tick = time.monotonic()
    images, actions, text = pixels(row)
    used = bind_generator_condition(pipe, row)
    frames, trace, pred, cache = candidates(pipe, row, seed, horizon=10)
    torch.cuda.synchronize()
    candidate_seconds = time.monotonic() - tick
    token_hash = hashlib.sha256(torch.stack([f['response'] for f in frames]).detach().cpu().numpy().tobytes()).hexdigest()
    metric_start = time.monotonic()
    metric_records = metric_reward(pred, images[2:12])
    torch.cuda.synchronize()
    metric_seconds = time.monotonic() - metric_start
    mr = torch.tensor([v['reward'] for v in metric_records], device='cuda:0')
    request_id = (batch - 1) * 2 + micro + 1
    request = R / f'group_{batch:04d}_micro{micro}.npz'
    assert not request.exists()
    np.savez_compressed(request, initial=images[1], videos=np.stack([as_video(v) for v in pred]), instruction=np.array(text))
    request_sha = sha(request)
    judge_start = time.monotonic()
    answer = judge.score(request_id, request)
    judge_seconds = time.monotonic() - judge_start
    assert answer['source_sha256'] == request_sha and len(answer['pairs']) == 6
    assert {(v['i'], v['j']) for v in answer['pairs']} == {(i, j) for i in range(4) for j in range(i + 1, 4)}
    for pair in answer['pairs']:
        for side in ('forward', 'reverse'):
            probs = pair[side]
            assert set(probs) == set('ABTU') and all(math.isfinite(v) for v in probs.values())
            assert abs(sum(probs.values()) - 1) < 1e-5
        raw_prefix = R / 'judgments' / f'g{request_id:04d}_p{pair["i"]}{pair["j"]}'
        assert Path(str(raw_prefix) + '_f.json').is_file() and Path(str(raw_prefix) + '_b.json').is_file()
    response_path = request.with_suffix('.reward.json')
    response_sha = sha(response_path)
    semantic = torch.tensor(answer['scores'], device='cuda:0')
    reward = mix(mr, semantic)
    assert reward.shape == (4,) and torch.isfinite(reward).all()
    record = dict(batch=batch, rank=RANK, micro=micro, virtual_rank=virtual_rank, seed=seed,
        episode=row['id'], prepared_input_sha256=row['sha256'], generator_condition=dict(text_mode='full',
        original_instruction=text, used_generator_instruction=used, generator_text_enabled=True),
        protocol_sha256=PROTOCOL_SHA256, metrics=metric_records, semantic=semantic.tolist(),
        candidate_seconds=candidate_seconds, metric_seconds=metric_seconds, judge_seconds=judge_seconds,
        rewards=reward.tolist(), reward_std=float(reward.std()), local_signal=int(float(reward.std()) > 1e-6),
        proposal_tokens_sha256=token_hash, judge_request_id=request_id, judge_request_sha256=request_sha,
        judge_response_sha256=response_sha, trace=trace, preparation_seconds=time.monotonic() - tick)
    del pred, images, mr, semantic
    gc.collect()
    torch.cuda.empty_cache()
    return dict(frames=frames, cache=cache, reward=reward, record=record, request=request, row=row)


try:
    status(state='INITIALIZING', updates=updates, target_updates=500, resume=resume_info)
    metric_reward = Metric()
    judge = Judge(R)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    training_start = time.monotonic()
    smoke_records = []
    target = 1 if args.smoke else 500
    max_batch = 1 if args.smoke else P['max_global_batches']
    for batch in range(first_batch, max_batch + 1):
        if updates == target:
            break
        tick = time.monotonic()
        opt.zero_grad(set_to_none=True)
        groups = [make_group(batch, micro) for micro in range(2)]
        local_signals = sum(g['record']['local_signal'] for g in groups)
        local_ties += 2 - local_signals
        count = torch.tensor(local_signals, device='cuda:0')
        dist.all_reduce(count)
        active = int(count)
        losses = []
        grad = None
        if active:
            for micro, group in enumerate(groups):
                cache = group['cache']
                actor.set_text(cache['features'], cache['mask'])
                reference.set_text(cache['features'], cache['mask'])
                def sync_context(t, micro=micro):
                    return forward.no_sync() if micro == 0 or t < 9 else contextlib.nullcontext()
                losses.append(policy_backward(actor, forward, reference, group['frames'], group['reward'],
                              cond(cache), 10, P['clip'], P['kl'], scale=.5, sync_context=sync_context))
            grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1, error_if_nonfinite=True))
            assert grad > 0 and not any(p.grad is not None for p in reference.parameters())
            opt.step()
            updates += 1
        else:
            global_skipped += 1
        torch.cuda.synchronize()
        last_batch = batch
        for micro, group in enumerate(groups):
            record = group['record']
            record.update(updates=updates, active_trajectories=active, grad_norm=grad,
                          loss=losses[micro] if losses else None, seconds=time.monotonic() - tick,
                          seconds_definition='Shared physical-rank global-batch wall time; do not sum over micro records',
                          requests_retained=False)
            durable_jsonl(R / 'train.jsonl', record)
            # Delete only after validated complete pairwise results and durable record.
            assert sha(group['request']) == record['judge_request_sha256']
            group['request'].unlink()
            if args.smoke:
                smoke_records.append(record)
        durable_jsonl(R / 'batches.jsonl', dict(batch=batch, updates=updates, active_trajectories=active,
              grad_norm=grad, seconds=time.monotonic() - tick, elapsed_train_seconds=elapsed_train(),
              local_ties=local_ties, global_skipped=global_skipped))
        del groups
        gc.collect()
        status(state='TRAINING', global_batches=batch, updates=updates, target_updates=target,
               global_skipped=global_skipped, local_ties=local_ties, elapsed_seconds=elapsed_train())
        print('DDP_STEP', RANK, batch, updates, active, grad, flush=True)
        if active and (args.smoke or updates in P['checkpoints']):
            checkpoint(updates, batch)
        if args.smoke or updates == target:
            break
    assert updates == target, ('Incomplete budget', updates, last_batch)
    peak_train = max(inherited_peak, float(torch.cuda.max_memory_allocated() / 2**30))
    training_seconds = elapsed_train()
    assert digest(reference) == initial
    judged = judge.close()
    judge = None
    del metric_reward
    final = agree(actor)
    assert final != initial
    if args.smoke:
        row = sel['smoke'][RANK]
        images, actions, text = pixels(row)
        used = bind_generator_condition(pipe, row)
        result = pipe.predict(images[:2].copy(), actions[:11].copy(), used, horizon=10, beam_width=4, memory='full')
        assert result['prediction'].shape == (10, 256, 320, 3) and np.isfinite(result['prediction']).all()
        np.savez_compressed(R / 'after_update.npz', prediction=result['prediction'], ground_truth=images[2:12], instruction=np.array(text))
        cp = O / 'checkpoint_000001'
        actor.core.load_state_dict(load_file(str(cp / 'model.safetensors'), device='cuda:0'), strict=True)
        actor.adapters.load_state_dict(load_file(str(cp / 'text_adapters.safetensors'), device='cuda:0'), strict=True)
        assert digest(actor) == final
        witnesses = gather(dict(rank=RANK, initial_digest=initial, final_digest=final, gradient=grad,
             checkpoint_reload_ok=True, records=smoke_records, judge=judged,
             seconds=training_seconds, reference_unchanged=True))
        if RANK == 0:
            assert len({v['final_digest'] for v in witnesses}) == 1
            assert sorted(v['virtual_rank'] for w in witnesses for v in w['records']) == list(range(8))
            receipt = dict(state='PASS', world=4, global_batch=8, gradient_accumulation=2,
                initial_digest=initial, final_digest=final, updates=1, protocol_sha256=PROTOCOL_SHA256,
                witnesses=witnesses, seconds=time.monotonic() - start,
                scope='Four-rank DDP, eight examples, immutable proposal/policy histories, complete hybrid reward, atomic checkpoint and inference; formal reloads original SFT')
            # This proves mechanics only. Runner writes E/gate.json after auditing
            # adaptive-tree diagnostics and any additional deployment checks.
            write(O / 'smoke_complete.json', receipt)
        status(state='SUCCEEDED', updates=1)
    else:
        del forward, reference, refpipe, opt
        gc.collect()
        torch.cuda.empty_cache()
        actor.eval().requires_grad_(False)
        metric = Metrics('cuda:0')
        evaluations = []
        eval_start = time.monotonic()
        for i, row in enumerate(sel['dev']):
            if i % WORLD != RANK:
                continue
            setup(20261011 + 10000 + i)
            images, actions, text = pixels(row)
            tick = time.monotonic()
            used = bind_generator_condition(pipe, row)
            result = pipe.predict(images[:2].copy(), actions[:33].copy(), used, horizon=32, beam_width=4, memory='full')
            prediction = np.asarray(result['prediction'], dtype='float32')
            gt = images[2:34]
            torch.cuda.synchronize()
            infer = time.monotonic() - tick
            scored = metric(prediction, gt, [10, 20, 32])
            if not evaluations:
                ident = metric(gt.astype('float32') / 255, gt, [32])['metrics']['32']
                assert abs(ident['ssim'] - 1) < 1e-5 and abs(ident['lpips']) < 1e-5
            path = R / f'dev_{i:03d}.npz'
            np.savez_compressed(path, prediction=prediction, tokens=result['tokens'], ground_truth=gt,
                                observations=images[:2], instruction=np.array(text))
            record = dict(id=row['id'], index=i, protocol_sha256=PROTOCOL_SHA256,
                initial_digest=initial, final_digest=final, metrics=scored['metrics'], per_frame=scored['per_frame'],
                frame32={k: v[-1] for k, v in scored['per_frame'].items()}, inference_seconds=infer,
                seconds=time.monotonic() - tick, artifact_path=str(path), artifact_sha256=sha(path),
                prepared_input_sha256=row['sha256'], generator_condition=dict(text_mode='full',
                original_instruction=text, used_generator_instruction=used, generator_text_enabled=True))
            write(R / f'dev_{i:03d}.json', record)
            evaluations.append(record)
            status(state='EVALUATING', updates=updates, cases=len(evaluations), total_cases=32)
            print('DDP_EVAL', RANK, len(evaluations), flush=True)
        assert len(evaluations) == 32 and digest(actor) == final
        own = dict(state='SUCCEEDED', rank=RANK, protocol_sha256=PROTOCOL_SHA256,
            initial_digest=initial, final_digest=final, updates=updates, global_batches=last_batch,
            global_skipped=global_skipped, local_ties=local_ties, peak_train_allocated_GiB=peak_train,
            train_seconds=training_seconds, eval_seconds=time.monotonic() - eval_start,
            evaluation=evaluations, judge=judged, resume=resume_info)
        write(R / 'complete.json', own)
        parts = gather(own)
        if RANK == 0:
            results = sorted([v for part in parts for v in part['evaluation']], key=lambda v: v['index'])
            assert [v['index'] for v in results] == list(range(128)) and len({v['id'] for v in results}) == 128
            base = json.loads(Path(P['baseline_complete']).read_text())
            assert [v['id'] for v in results] == [v['id'] for v in base['evaluation']]
            assert all(a['prepared_input_sha256'] == b['prepared_input_sha256'] for a, b in zip(results, base['evaluation']))
            assert base['initial_digest'] == initial
            assert len({v['final_digest'] for v in parts}) == 1 and all(v['updates'] == 500 for v in parts)
            mean32 = {key: float(np.mean([row['metrics']['32'][key] for row in results])) for key in results[0]['metrics']['32']}
            last = {key: float(np.mean([row['frame32'][key] for row in results])) for key in results[0]['frame32']}
            train_seconds_max = max(v['train_seconds'] for v in parts)
            final_checkpoint = O / 'checkpoint_000500'
            if not final_checkpoint.exists():
                assert args.resume and resume_info['resumed_from_update'] == 500
                final_checkpoint = args.resume.resolve()
            final_receipt = json.loads((final_checkpoint / 'receipt.json').read_text())
            assert final_receipt['updates'] == 500 and final_receipt['digest'] == final
            write(O / 'complete.json', dict(state='SUCCEEDED', arm=ARM, text_mode='full',
                protocol_sha256=PROTOCOL_SHA256, world_size=4, global_batch=8, gradient_accumulation=2,
                seed=20261011, updates=updates, groups=last_batch * 8, global_batches=last_batch,
                skipped=global_skipped, local_ties_total=sum(v['local_ties'] for v in parts),
                initial_digest=initial, final_digest=final, mean_frames1_32=mean32, frame32_only=last,
                evaluation=results, train_seconds=train_seconds_max, train_gpu_hours=train_seconds_max * 4 / 3600,
                train_seconds_definition='Observed allocated four-GPU training stage including proposals, hybrid scoring, policy updates and checkpoints; excludes loading, smoke and final evaluation. Resume adds retained checkpoint-prefix time.',
                eval_seconds=max(v['eval_seconds'] for v in parts), protocol=P, scope=P['scope'], source=source,
                resume=resume_info, final_checkpoint_path=str(final_checkpoint),
                final_checkpoint_receipt_sha256=sha(final_checkpoint / 'receipt.json'),
                peak_train_allocated_GiB=max(v['peak_train_allocated_GiB'] for v in parts)))
            print('DDP_COMPLETE', json.dumps(mean32), flush=True)
        status(state='SUCCEEDED', updates=updates, cases=32)
    verify_assets()
    dist.barrier()
except BaseException:
    write(R / 'ERROR.json', dict(error=traceback.format_exc(), rank=RANK))
    raise
finally:
    if judge:
        judge.abort()
    if dist.is_initialized():
        dist.destroy_process_group()
