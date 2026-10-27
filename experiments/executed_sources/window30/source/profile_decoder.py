"""Instrument the frozen window decoder without changing its mathematical path.

``predict_profiled`` has the same arguments as ``predict_with_window`` and returns
``(result, profile)``. The result retains exactly the original output schema.
Only the two outer profiling boundaries explicitly synchronize CUDA. Per-frame
events are read after the final synchronization; they do not add frame barriers.

Package the original window_decoder.py beside this file, or set
R2_WINDOW_DECODER_PATH to its immutable path. Its complete source hash is checked
before import. The relative fallback supports this local workspace only.

Caller responsibilities: warm up every measured configuration, freeze FP32/TF32,
preload/cache text if text-encoder startup is outside the chosen timing scope,
exclude scoring models from the inference process, verify context admission,
and require real-GPU token/pixel parity before accepting measurements.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import time
from typing import Any

import torch


WINDOW_DECODER_SHA256 = "d9bdd14d292ac44afe3e45ba0f2d3d6fe61c9c246147b43e98afb12b21acf150"


def _load_frozen_decoder():
    explicit = os.environ.get("R2_WINDOW_DECODER_PATH")
    here = Path(__file__).resolve().parent
    candidates = ([Path(explicit)] if explicit else [
        here / "window_decoder.py",
        here.parent / "r2_long_memory_20261024/decoder/window_decoder.py",
    ])
    source = next((path.resolve() for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError("Frozen window decoder not found; set R2_WINDOW_DECODER_PATH")
    source_bytes = source.read_bytes()
    actual = hashlib.sha256(source_bytes).hexdigest()
    if actual != WINDOW_DECODER_SHA256:
        raise ValueError(f"Frozen window decoder SHA mismatch: {source}: {actual}")
    spec = importlib.util.spec_from_file_location("_r2_frozen_window_decoder", source)
    module = importlib.util.module_from_spec(spec)
    # Compile the bytes just checked, rather than asking the loader to reread a
    # path that could have changed between checksum validation and execution.
    exec(compile(source_bytes, str(source), "exec"), module.__dict__)
    return module, source


_window, WINDOW_DECODER_PATH = _load_frozen_decoder()


def source_audit():
    """Identity of the original source loaded into this profiling module."""
    return {
        "window_decoder_path": str(WINDOW_DECODER_PATH),
        "window_decoder_sha256": WINDOW_DECODER_SHA256,
        "profile_decoder_path": str(Path(__file__).resolve()),
        "profile_decoder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


class _CudaProfiler:
    """CUDA stream elapsed times plus host wall times; no per-frame reset/barrier."""

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Profiling requires a CUDA pipeline; CPU fakes are structural tests only")
        self.stream = torch.cuda.current_stream(self.device)
        self.stages = {}
        self.frames = []

    def _event(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        return event

    def start(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.initial_allocated = torch.cuda.memory_allocated(self.device)
        self.initial_reserved = torch.cuda.memory_reserved(self.device)
        self.started = time.perf_counter()

    def stage_start(self, name):
        self.stages[name] = {"start_wall": time.perf_counter(), "start_event": self._event()}

    def stage_end(self, name):
        self.stages[name].update(end_event=self._event(), end_wall=time.perf_counter())

    def frame_start(self, target):
        self.frames.append({"target": target, "predicted_frame": target - 1,
                            "start_event": self._event()})

    def frame_end(self, prefix_length):
        self.frames[-1].update(
            end_event=self._event(), prefix_length=int(prefix_length),
            allocated_bytes_after_enqueue=torch.cuda.memory_allocated(self.device),
            reserved_bytes_after_enqueue=torch.cuda.memory_reserved(self.device),
            running_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
        )

    def finish(self):
        torch.cuda.synchronize(self.device)
        total_wall = time.perf_counter() - self.started
        stages = {
            name: {
                "host_wall_seconds": row["end_wall"] - row["start_wall"],
                "cuda_event_ms": row["start_event"].elapsed_time(row["end_event"]),
            }
            for name, row in self.stages.items()
        }
        frames = [{
            **{key: value for key, value in row.items() if not key.endswith("_event")},
            "cuda_event_ms": row["start_event"].elapsed_time(row["end_event"]),
        } for row in self.frames]
        peak = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        return {
            "version": 1,
            "device": str(self.device),
            "total_wall_seconds": total_wall,
            "stages": stages,
            "frames": frames,
            "initial_allocated_bytes": self.initial_allocated,
            "end_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "peak_allocated_bytes": peak,
            "peak_increment_above_initial_bytes": peak - self.initial_allocated,
            "initial_reserved_bytes": self.initial_reserved,
            "end_reserved_bytes": torch.cuda.memory_reserved(self.device),
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_increment_above_initial_bytes": peak_reserved - self.initial_reserved,
            "definitions": {
                "total_wall_seconds": "Synchronized wall time from initial encoding through output D2H, including instrumentation; excludes source audit, warmup and scoring.",
                "stages_host_wall_seconds": "Host elapsed/submission times without intermediate barriers; async stages are not independently synchronized latencies.",
                "cuda_event_ms": "Elapsed time between events on the pipeline device current stream, read only after final synchronization; includes any stream idle time, not a sum of kernel execution times.",
                "frame_cuda_event_ms": "Context/prefix construction, native beam decode, response bookkeeping and history pruning; excludes whole-video detokenization and output D2H.",
                "allocated_bytes_after_enqueue": "PyTorch allocator snapshot after enqueuing the frame end event; no per-frame synchronization.",
                "running_peak_allocated_bytes": "Cumulative PyTorch peak since inference start, including earlier stages and resident models; never a frame-independent peak. Reset occurs once per invocation.",
                "peak_allocated_bytes": "Cumulative inference PyTorch allocated peak, including whole-video detokenization; excludes non-PyTorch allocations and is not device-wide/reserved memory.",
                "reserved_bytes": "PyTorch caching allocator reservation, including reusable blocks from warmup/earlier cases. The same single reset clears allocated and reserved peak counters, but does not release cached storage. This is not device-wide or total process memory.",
                "initial_encoding": "Native encode_observations, action tokenization, condition(instruction), and model.eval; caller controls whether text was prewarmed.",
                "detokenize": "Exactly one native whole-video detokenize call; retained output memory can grow with horizon.",
            },
        }


@torch.no_grad()
def predict_profiled(
    pipeline,
    observations,
    actions,
    instruction="",
    horizon=128,
    beam_width=4,
    recent_states=6,
    retain_anchor=True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return native-equivalent results and separate CUDA profiling metadata.

    Event recording does not change decoding, random state, floating-point
    precision, cache reset, retained state/action indices, or detokenization.
    The original beam implementation can itself synchronize for its assertions
    and traces; this wrapper neither adds frame barriers nor removes native ones.
    Real-model GPU parity remains a required caller-side gate.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if len(actions) < horizon + 1:
        raise ValueError("Insufficient planned controls")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    _window.retained_indices(2, recent_states, retain_anchor)
    encode_actions, beam_decode, TensorBatch = _window._native_dependencies(pipeline)
    audit = source_audit()
    profiler = _CudaProfiler(pipeline.device)
    profiler.start()
    profiler.stage_start("initial_encoding")
    ctx, anchor = pipeline.encode_observations(observations)
    acts = encode_actions(
        torch.as_tensor(actions, device=pipeline.device), pipeline.ranges, pipeline.action_dim
    )
    pipeline.condition(instruction)
    pipeline.model.eval()
    profiler.stage_end("initial_encoding")
    profiler.stage_start("autoregressive")
    history = {1: anchor}
    responses = []
    traces = []
    for t in range(2, horizon + 2):
        profiler.frame_start(t)
        context = ctx.reshape(1, -1) + 4375
        prefix, kept = _window.make_window_prefix(
            context, history, acts, t, recent_states, retain_anchor
        )
        candidates, _, trace = beam_decode(pipeline.model, prefix, width=beam_width)
        history[t] = candidates[:1]
        responses.append(candidates[:1])
        history = _window.prune_history(history, t, recent_states, retain_anchor)
        traces.append(dict(trace, target=t, retained=kept, stored_after=list(history)))
        profiler.frame_end(prefix.shape[1])
    codes = torch.stack(responses, 1)
    profiler.stage_end("autoregressive")
    profiler.stage_start("detokenize")
    prediction = pipeline.worker.detokenize(
        TensorBatch({"tokens": codes, "ctx_tokens": ctx}), None
    ).batch["pixels"][:, 1:]
    profiler.stage_end("detokenize")
    profiler.stage_start("d2h")
    result = dict(
        tokens=codes.cpu().numpy(),
        prediction=prediction[0].permute(0, 2, 3, 1).cpu().numpy(),
        trace=traces,
        memory=dict(recent_states=recent_states, retain_anchor=retain_anchor,
                    fixed_initial_scene=True, updates=0),
    )
    profiler.stage_end("d2h")
    profile = profiler.finish()
    profile["source_audit"] = audit
    return result, profile
