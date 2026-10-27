"""Real four-process CPU/Gloo gate for the 4 x 2 accumulation schedule.

Run before the GPU smoke/formal experiment, for example:
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
      test_ddp_accum.py --source-root code/src

Uses the exact source policy_backward function (AST extraction avoids loading
the CUDA/data pipeline) and the original objective/frame_logp functions. A tiny
causal language model replaces only the expensive world model. This checks
gradients, clipped gradients, two AdamW steps and optimizer moments against a
sequential global-eight mean. It does not validate GPU numerics or MCTS quality.
"""
import argparse
import ast
import contextlib
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_actual_policy(source):
    memory = load_file_module('ddp_gate_memory', source / 'futureworlds/memory.py')
    objective = load_file_module('ddp_gate_objective', source / 'futureworlds/objective.py')
    path = source / 'futureworlds/training.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'policy_backward']
    assert len(nodes) == 1
    scope = dict(torch=torch, np=np, contextlib=contextlib, frame_logp=memory.frame_logp,
                 compute_grpo_outcome_advantage=objective.compute_grpo_outcome_advantage,
                 compute_policy_loss=objective.compute_policy_loss, kl_penalty=objective.kl_penalty)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['policy_backward'], memory


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(19, 7)
        self.middle = torch.nn.Linear(7, 9)
        self.output = torch.nn.Linear(9, 19)

    def forward(self, input_ids, use_cache=False, **unused):
        embedding = self.embedding(input_ids)
        positions = torch.arange(1, embedding.shape[1] + 1, dtype=embedding.dtype)[None, :, None]
        hidden = embedding.cumsum(dim=1) / positions
        return SimpleNamespace(logits=self.output(torch.tanh(self.middle(hidden))))


def groups_for(actor, memory, active, step):
    """Eight fixed groups, four candidates/group and ten 80-token frames/group."""
    result = []
    for index in range(8):
        generator = torch.Generator().manual_seed(31000 + 100 * step + index)
        context = torch.randint(0, 19, (4, 5), generator=generator)
        # Shared observed context/anchor/actions, branch-local generated states.
        context[:] = context[0].clone()
        anchor = torch.randint(0, 19, (1, 80), generator=generator).expand(4, -1)
        actions = torch.randint(0, 19, (12, 2), generator=generator)
        history, frames = {1: anchor}, []
        for target in range(2, 12):
            prefix, retained = memory.make_prefix(context, history, actions, target)
            response = torch.randint(0, 19, (4, 80), generator=generator)
            with torch.no_grad():
                saved = memory.frame_logp(actor, prefix, response)
            frames.append(dict(target=target, prefix=prefix, response=response, model_logp=saved))
            history[target] = response
        assert retained == [1, 5, 6, 7, 8, 9, 10]
        reward = torch.tensor([-0.9, 0.2, 0.5, 1.4]) if index in active else torch.ones(4)
        result.append((frames, reward))
    return result


def flattened_grad(model):
    parts = []
    for parameter in model.parameters():
        parts.append((parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1))
    return torch.cat(parts)


def flattened_parameters(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def close(actual, expected, name, *, atol=2e-6, rtol=2e-5):
    error = float((actual - expected).abs().max()) if actual.numel() else 0.0
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f'{name}: max_abs_error={error}')
    return error


def optimizer_moment_error(actual, expected):
    errors = {}
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        a = [state[key].detach().reshape(-1) for state in actual.state.values()]
        b = [state[key].detach().reshape(-1) for state in expected.state.values()]
        assert len(a) == len(b)
        errors[key] = close(torch.cat(a), torch.cat(b), 'optimizer/' + key) if a else 0.0
    return errors


def main():
    parser = argparse.ArgumentParser()
    default = Path(__file__).resolve().parent / 'code/src'
    if not default.exists():
        default = Path(__file__).resolve().parents[2] / 'FutureWorlds/src'
    parser.add_argument('--source-root', type=Path, default=default)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    policy_backward, memory = load_actual_policy(source)
    torch.set_num_threads(1)
    assert int(os.environ['WORLD_SIZE']) == 4
    dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    cases = [
        ('all_active', set(range(8))),
        ('rank0_one_active', set(range(8)) - {0}),
        ('rank0_zero_active', set(range(8)) - {0, 4}),
        ('only_one_global_active', {7}),
        ('all_tied_skip', set()),
    ]
    records = []
    try:
        for case_index, (name, active) in enumerate(cases):
            torch.manual_seed(20261025 + case_index)
            actor = TinyCausalLM().eval()
            serial = copy.deepcopy(actor)
            reference = copy.deepcopy(actor).requires_grad_(False)
            frozen = flattened_parameters(reference).clone()
            forward = DDP(actor, broadcast_buffers=False, gradient_as_bucket_view=True)
            optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, weight_decay=0)
            serial_optimizer = torch.optim.AdamW(serial.parameters(), lr=1e-3, weight_decay=0)
            initial = flattened_parameters(actor).clone()
            for step in range(2):
                groups = groups_for(actor, memory, active, step)
                optimizer.zero_grad(set_to_none=True)
                serial_optimizer.zero_grad(set_to_none=True)
                local_signals = sum(rank + micro * world in active for micro in range(2))
                signal_count = torch.tensor(local_signals)
                dist.all_reduce(signal_count)
                assert int(signal_count) == len(active)
                schedule, losses = [], []
                if active:
                    # Independent local-gradient witness, before all-reduce.
                    local = copy.deepcopy(actor)
                    for micro in range(2):
                        frames, reward = groups[rank + micro * world]
                        policy_backward(local, local, reference, frames, reward, {}, 10, .2, .001, scale=.5)
                    local_norm = float(flattened_grad(local).norm())
                    if step == 0 and local_signals == 0:
                        assert local_norm == 0.0, ('Expected exactly zero local first-step gradient', name, rank, local_norm)
                    del local
                    for micro in range(2):
                        frames, reward = groups[rank + micro * world]
                        def sync_context(t, micro=micro):
                            sync = micro == 1 and t == 9
                            schedule.append(dict(micro=micro, frame=t, sync=sync))
                            return contextlib.nullcontext() if sync else forward.no_sync()
                        losses.append(policy_backward(actor, forward, reference, frames, reward,
                                                      {}, 10, .2, .001, scale=.5, sync_context=sync_context))
                    assert len(schedule) == 20 and sum(v['sync'] for v in schedule) == 1 and schedule[-1]['sync']
                    for frames, reward in groups:
                        policy_backward(serial, serial, reference, frames, reward, {}, 10, .2, .001, scale=1 / 8)
                    gradient_error = close(flattened_grad(actor), flattened_grad(serial), name + '/raw_gradient')
                    norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1, error_if_nonfinite=True))
                    serial_norm = float(torch.nn.utils.clip_grad_norm_(serial.parameters(), 1, error_if_nonfinite=True))
                    clipped_error = close(flattened_grad(actor), flattened_grad(serial), name + '/clipped_gradient')
                    optimizer.step()
                    serial_optimizer.step()
                    parameter_error = close(flattened_parameters(actor), flattened_parameters(serial), name + '/parameters')
                    moment_error = optimizer_moment_error(optimizer, serial_optimizer)
                    assert norm > 0 and abs(norm - serial_norm) < 2e-6
                else:
                    local_norm = norm = 0.0
                    gradient_error = clipped_error = parameter_error = 0.0
                    moment_error = {}
                    assert not optimizer.state and not serial_optimizer.state
                    assert torch.equal(flattened_parameters(actor), initial)
                    assert all(p.grad is None for p in actor.parameters())
                assert torch.equal(flattened_parameters(reference), frozen)
                assert all(p.grad is None for p in reference.parameters())
                all_parameters = [torch.empty_like(initial) for _ in range(world)]
                dist.all_gather(all_parameters, flattened_parameters(actor))
                assert all(torch.equal(all_parameters[0], value) for value in all_parameters[1:])
                local_records = [None] * world
                dist.all_gather_object(local_records, dict(rank=rank, local_active=local_signals,
                    local_pre_reduce_norm=local_norm, global_norm=norm, raw_gradient_max_error=gradient_error,
                    clipped_gradient_max_error=clipped_error, parameter_max_error=parameter_error,
                    moment_max_errors=moment_error, synchronization_calls=sum(v['sync'] for v in schedule)))
                records.append(dict(case=name, step=step + 1, global_active=len(active),
                                    optimizer_step=bool(active), ranks=local_records))
            del forward, actor, serial, reference, optimizer, serial_optimizer
            dist.barrier()
        if rank == 0:
            receipt = dict(state='PASS', backend='gloo', device='cpu', world_size=4,
                accumulation=2, trajectory_global_batch=8, horizon=10, candidates_per_trajectory=4,
                per_trajectory_scale=.5, sequential_scale=1 / 8, toy_optimizer_lr=.001,
                scope='Real DDP gradient/AdamW equivalence only; not a CUDA or search-quality gate',
                torch=torch.__version__, source_root=str(source),
                source_sha256={name: sha(source / 'futureworlds' / name)
                               for name in ('training.py', 'memory.py', 'objective.py')},
                test_sha256=sha(__file__), cases=records)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            print(json.dumps(receipt, allow_nan=False), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
