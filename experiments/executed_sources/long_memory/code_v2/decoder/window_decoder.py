"""Audited, inference-only window extension of WorldPipeline.predict.

The world model, action tokenizer, per-frame beam decoder and video detokenizer
remain native.  Only retained history indices and matching history pruning are
parameterized.  No future video, checkpoint loader, optimizer or network API is
accepted here.  This file is independent of the archived experiment sources.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch


ARMS = {
    "full_w1": {"recent_states": 1, "retain_anchor": True},
    "full_w3": {"recent_states": 3, "retain_anchor": True},
    "full_w6": {"recent_states": 6, "retain_anchor": True},
    "full_w12": {"recent_states": 12, "retain_anchor": True},
    "recent_w6": {"recent_states": 6, "retain_anchor": False},
    "current_only": {"recent_states": 1, "retain_anchor": False},
}


def retained_indices(target: int, recent_states: int, retain_anchor: bool) -> list[int]:
    """State k is followed by action[k], which predicts the next dynamic state.

The fixed first dynamic state is not counted in the rolling window after it
ages out of that window.  Thus full_w6 has at most seven visible states, whereas
recent_w6 has at most six.  The scene prefix is present in both conditions.
"""
    if isinstance(target, bool) or not isinstance(target, int) or target < 2:
        raise ValueError("target must be an integer >= 2")
    if isinstance(recent_states, bool) or not isinstance(recent_states, int) or recent_states < 1:
        raise ValueError("recent_states must be a positive integer")
    if not isinstance(retain_anchor, bool):
        raise ValueError("retain_anchor must be Boolean")
    if retain_anchor:
        return [1] + list(range(max(2, target - recent_states), target))
    return list(range(max(1, target - recent_states), target))


def make_window_prefix(context, history, actions, target, recent_states=6, retain_anchor=True):
    """Concatenate exactly the native scene/state/action layout, without a KV cache."""
    kept = retained_indices(target, recent_states, retain_anchor)
    if any(k not in history for k in kept):
        raise ValueError(f"Missing retained state at target {target}: {kept}")
    if max(kept) >= actions.shape[0]:
        raise ValueError("Insufficient tokenized actions")
    blocks = [context]
    for k in kept:
        blocks += [history[k], actions[k].expand(context.shape[0], -1)]
    return torch.cat(blocks, dim=1), kept


def prune_history(history, completed_target, recent_states=6, retain_anchor=True):
    """Keep precisely what the following frame can read; never drop the anchor."""
    next_kept = set(retained_indices(completed_target + 1, recent_states, retain_anchor))
    return {k: v for k, v in history.items() if k in next_kept}


def _native_dependencies(pipeline):
    # Obtain the actual implementation's own helpers rather than importing a
    # possibly different installed futureworlds package. torch.no_grad uses
    # functools.wraps, so inspect.unwrap reaches the archived predict function.
    native = inspect.unwrap(pipeline.predict)
    namespace = native.__globals__
    names = ("encode_actions", "beam_decode", "TensorBatch")
    if any(name not in namespace for name in names):
        raise TypeError("pipeline.predict does not expose the audited native helpers")
    return tuple(namespace[name] for name in names)


@torch.no_grad()
def predict_with_window(
    pipeline,
    observations,
    actions,
    instruction="",
    horizon=128,
    beam_width=4,
    recent_states=6,
    retain_anchor=True,
) -> dict[str, Any]:
    """Run fixed-checkpoint beam inference with a configurable dynamic history.

For recent_states=6/retain_anchor=True this has exactly the native full-memory
prefixes, pruning and whole-video detokenization order.  H=64/128 changes only
the count of repeated autoregressive steps.  Caller must verify actual action
and GT availability; no padding, action repetition or horizon truncation occurs.
The caller is also responsible for model context-budget admission and GPU parity.
"""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if len(actions) < horizon + 1:
        raise ValueError("Insufficient planned controls")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    retained_indices(2, recent_states, retain_anchor)  # validate before GPU work
    encode_actions, beam_decode, TensorBatch = _native_dependencies(pipeline)
    ctx, anchor = pipeline.encode_observations(observations)
    acts = encode_actions(
        torch.as_tensor(actions, device=pipeline.device), pipeline.ranges, pipeline.action_dim
    )
    pipeline.condition(instruction)
    pipeline.model.eval()
    history = {1: anchor}
    responses = []
    traces = []
    for t in range(2, horizon + 2):
        context = ctx.reshape(1, -1) + 4375
        prefix, kept = make_window_prefix(context, history, acts, t, recent_states, retain_anchor)
        candidates, _, trace = beam_decode(pipeline.model, prefix, width=beam_width)
        history[t] = candidates[:1]
        responses.append(candidates[:1])
        history = prune_history(history, t, recent_states, retain_anchor)
        traces.append(dict(trace, target=t, retained=kept, stored_after=list(history)))
    codes = torch.stack(responses, 1)
    prediction = pipeline.worker.detokenize(
        TensorBatch({"tokens": codes, "ctx_tokens": ctx}), None
    ).batch["pixels"][:, 1:]
    return dict(
        tokens=codes.cpu().numpy(),
        prediction=prediction[0].permute(0, 2, 3, 1).cpu().numpy(),
        trace=traces,
        memory=dict(recent_states=recent_states, retain_anchor=retain_anchor,
                    fixed_initial_scene=True, updates=0),
    )


def validate_trace(trace, *, horizon, recent_states, retain_anchor):
    """Fail closed on missing frames, state/action mis-indexing or prune errors."""
    if len(trace) != horizon:
        raise ValueError(f"Expected {horizon} steps, got {len(trace)}")
    for target, row in enumerate(trace, 2):
        expected = retained_indices(target, recent_states, retain_anchor)
        stored = retained_indices(target + 1, recent_states, retain_anchor)
        if row["target"] != target or row["retained"] != expected or row["stored_after"] != stored:
            raise ValueError(f"Invalid rolling history at target {target}: {row}")
    return True
