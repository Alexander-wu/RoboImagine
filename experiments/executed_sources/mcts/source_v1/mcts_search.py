"""Frame-prefix UCT using likelihood-only rollouts, for an isolated RT-1 baseline.

This is Monte-Carlo tree search, not independent sampling renamed as MCTS:
existing frame-prefix nodes are revisited by UCT, one node is expanded per
simulation when possible, and each complete rollout backs up its value to every
node on the selected tree path. A terminal tree revisit also counts as a complete
simulation. Duplicates are legal and are never replaced to manufacture diversity.

Terminal value is exp(mean raw model token log probability). The current observed
terminal-value min/max rescales node mean values for UCT selection only; c_uct=1
is fixed, not tuned. Final trajectories are selected by raw cumulative model log
probability. Neither visual ground truth nor VLM feedback enters this search.

The default 16 simulations have a different and larger nominal rollout budget
than the eight-path beam control. Actual forward/token-row counters must be
reported, rather than describing this experiment as compute matched. The search
proposal distribution is not the autoregressive policy's sampling distribution.
"""
from dataclasses import dataclass, field
import math
import hashlib
from typing import List, Optional

import torch
from futureworlds.memory import make_prefix


@dataclass
class _Node:
    node_id: int
    depth: int
    parent: Optional[int]
    frames: list
    children: List[int] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self):
        return self.value_sum / self.visits if self.visits else 0.0


def _normalized_q(value, value_min, value_max):
    """Neutral value until a non-degenerate range has actually been observed."""
    if value_min is None or value_max - value_min <= 1e-15:
        return 0.5
    return min(1.0, max(0.0, (value - value_min) / (value_max - value_min)))


def _choose_child(node, nodes, value_min, value_max, c_uct):
    """Unvisited first; then stable (creation-order) ties between UCT scores."""
    alternatives = []
    for child_id in node.children:
        child = nodes[child_id]
        exploitation = _normalized_q(child.mean_value, value_min, value_max)
        exploration = (
            c_uct * math.sqrt(math.log(node.visits + 1.0) / child.visits)
            if child.visits else None
        )
        alternatives.append(dict(
            node_id=child_id, visits=child.visits, mean_value=child.mean_value,
            exploitation=exploitation, exploration=exploration,
            uct=(exploitation + exploration if exploration is not None else None),
        ))
    unvisited = [a for a in alternatives if not a['visits']]
    chosen = unvisited[0] if unvisited else max(alternatives, key=lambda a: a['uct'])
    return chosen['node_id'], dict(
        parent_node=node.node_id, parent_visits=node.visits,
        value_min=value_min, value_max=value_max,
        selected_node=chosen['node_id'], unvisited_first=bool(unvisited),
        alternatives=alternatives,
    )


def _history(anchor, frames):
    return {1: anchor, **{frame['target']: frame['response'] for frame in frames}}


def _check_prefixes(context, anchor, actions, frames, recent):
    history = _history(anchor, frames)
    checks = []
    for frame in frames:
        rebuilt, _ = make_prefix(context, history, actions, frame['target'], recent=recent)
        checks.append((rebuilt == frame['prefix']).all())
    if checks and not bool(torch.stack(checks).all()):
        raise AssertionError('MCTS branch ancestry/prefix mismatch')
    return len(frames)


def _score(frames):
    # Same unmodified token log probabilities consumed by the policy surrogate.
    # One device-to-host scalar transfer per trajectory, not one per frame.
    return float(sum(frame['model_logp'].double().sum() for frame in frames)) if frames else 0.0


@torch.no_grad()
def rollout_mcts(
    model, context, anchor, actions, horizon, generator, *,
    simulations=16, branching=2, returned=4, visual=4375, width=80,
    recent=6, c_uct=1.0, temperature=1.0,
):
    """Return beam-compatible batched frame records and a JSON-safe audit trace.

    Caller binds text with model.set_text before calling, as for beam controls.
    All tree edges represent sampled future visual frames, not action decisions.
    Conditions/actions remain fixed. Per-frame KV caches are reset; within-frame
    token sampling reuses KV. The observed dynamic anchor is retained unchanged.
    """
    if min(horizon, simulations, branching, returned, visual, width, recent) < 1:
        raise ValueError('All counts must be positive')
    if returned > simulations:
        raise ValueError('Cannot return more simulation records than completed simulations')
    if context.ndim != 2 or context.shape[0] != 1 or anchor.shape != (1, width):
        raise ValueError('Expected context [1,C] and anchor [1,width]')
    if actions.ndim != 2 or actions.shape[0] < horizon + 1:
        raise ValueError('Insufficient aligned actions for all predicted states')
    if temperature != 1.0:
        raise ValueError('This sealed likelihood baseline requires temperature=1')
    if not math.isfinite(c_uct) or c_uct < 0:
        raise ValueError('c_uct must be finite and nonnegative')
    if generator is None:
        raise ValueError('An explicitly seeded torch.Generator is required')
    model.eval()
    counters = dict(
        model_forward_calls=0, actor_token_logit_rows=0,
        produced_logit_rows=0, input_token_rows=0,
        generated_token_rows=0, generated_frames=0,
        expansion_frames=0, rollout_frames=0,
    )

    def sample_frame(previous_frames, phase):
        target = len(previous_frames) + 2
        prefix, _ = make_prefix(
            context, _history(anchor, previous_frames), actions, target, recent=recent,
        )
        pending, kv = prefix, None
        codes, logps = [], []
        for _ in range(width):
            counters['model_forward_calls'] += 1
            counters['input_token_rows'] += pending.numel()
            out = model(input_ids=pending, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            counters['produced_logit_rows'] += out.logits.shape[0] * out.logits.shape[1]
            logp = out.logits[:, -1, :visual].float().log_softmax(-1)
            if logp.shape != (1, visual):
                raise RuntimeError('Invalid visual-token logit shape')
            pending = torch.multinomial(logp.exp(), 1, generator=generator)
            codes.append(pending)
            logps.append(logp.gather(1, pending))
            counters['actor_token_logit_rows'] += 1
            counters['generated_token_rows'] += 1
        counters['generated_frames'] += 1
        counters[phase + '_frames'] += 1
        response = torch.cat(codes, dim=1).detach()
        saved_logp = torch.cat(logps, dim=1).detach()
        if not bool(torch.isfinite(saved_logp).all()):
            raise RuntimeError('Non-finite sampled model token log probability')
        return dict(target=target, prefix=prefix.detach(),
                    response=response, model_logp=saved_logp)

    nodes = [_Node(0, 0, None, [])]
    terminal_min = terminal_max = None
    candidates, simulation_traces = [], []
    prefix_checks = 0
    for sim in range(simulations):
        node = nodes[0]
        path, selections = [0], []
        while node.depth < horizon and len(node.children) == branching:
            child_id, choice = _choose_child(node, nodes, terminal_min, terminal_max, c_uct)
            selections.append(choice)
            node = nodes[child_id]
            path.append(child_id)
        expanded_id = None
        terminal_revisit = node.depth == horizon
        if not terminal_revisit:
            frame = sample_frame(node.frames, 'expansion')
            expanded_id = len(nodes)
            child = _Node(expanded_id, node.depth + 1, node.node_id, node.frames + [frame])
            nodes.append(child)
            node.children.append(expanded_id)
            node = child
            path.append(expanded_id)
        full_frames = list(node.frames)
        while len(full_frames) < horizon:
            full_frames.append(sample_frame(full_frames, 'rollout'))
        prefix_checks += _check_prefixes(context, anchor, actions, full_frames, recent)
        token_bytes = torch.cat([f['response'] for f in full_frames], 1).cpu().numpy().astype('<i8', copy=False).tobytes(order='C')
        token_sha256 = hashlib.sha256(token_bytes).hexdigest()
        raw_score = _score(full_frames)
        mean_logp = raw_score / (horizon * width)
        value = math.exp(mean_logp)
        if not math.isfinite(value):
            raise RuntimeError('Non-finite MCTS terminal value')
        terminal_min = value if terminal_min is None else min(terminal_min, value)
        terminal_max = value if terminal_max is None else max(terminal_max, value)
        for node_id in path:
            nodes[node_id].visits += 1
            nodes[node_id].value_sum += value
        candidates.append(dict(frames=full_frames, raw_score=raw_score, simulation=sim))
        simulation_traces.append(dict(
            simulation=sim, selected_tree_path=path, selections=selections,
            expanded_node=expanded_id, selected_depth=node.depth,
            terminal_revisit=terminal_revisit, raw_model_log_score=raw_score,
            mean_model_token_logp=mean_logp, terminal_value=value,
            trajectory_token_sha256=token_sha256,
            value_min_after=terminal_min, value_max_after=terminal_max,
            backed_up_nodes=list(path),
        ))
    # Python's sort is stable, including equal scores or duplicate trajectories.
    order = sorted(range(simulations), key=lambda i: -candidates[i]['raw_score'])
    selected = order[:returned]
    frames = [dict(
        target=target + 2,
        **{key: torch.cat([candidates[i]['frames'][target][key] for i in selected], dim=0).clone()
           for key in ['prefix', 'response', 'model_logp']},
    ) for target in range(horizon)]
    history = {1: anchor.expand(returned, -1), **{f['target']: f['response'] for f in frames}}
    frame_traces = []
    for f in frames:
        rebuilt, kept = make_prefix(
            context.expand(returned, -1), history, actions, f['target'], recent=recent,
        )
        if not torch.equal(rebuilt, f['prefix']):
            raise AssertionError('Returned MCTS trajectory has mismatched branch history')
        frame_traces.append(dict(
            target=f['target'], retained=kept, prefix_length=f['prefix'].shape[1],
            positions_reset=True, kv_reset=True, future_observations=False,
            random_sampling=True, selected_simulation_indices=list(selected),
        ))
    raw = torch.tensor([candidates[i]['raw_score'] for i in selected], dtype=torch.float64, device=context.device)
    reconstructed = sum(f['model_logp'].double().sum(-1) for f in frames)
    score_error = float((raw - reconstructed).abs().max())
    if not torch.allclose(reconstructed, raw, atol=1e-7, rtol=0):
        raise AssertionError('Raw MCTS policy log-score is inconsistent with frame records')
    codes = torch.stack([f['response'] for f in frames], dim=1)
    pair = [float((codes[i] != codes[j]).float().mean())
            for i in range(returned) for j in range(i + 1, returned)]
    different_per_frame = (codes != codes[:1]).any(0).sum(-1).tolist()
    return frames, dict(
        strategy='likelihood_guided_frame_prefix_uct_mcts',
        simulations=simulations, branching=branching, c_uct=c_uct,
        terminal_value='exp(mean raw model token log probability)',
        selection_value_normalization='online observed terminal minmax; neutral 0.5 if degenerate',
        selection_ties='unvisited first, then node creation order',
        final_selection='stable top raw cumulative model log probability',
        returned_trajectories=returned, selected_simulation_indices=selected,
        duplicate_simulations_retained=True,
        generated_trajectory_records=simulations,
        all_simulation_token_sha256=[s['trajectory_token_sha256'] for s in simulation_traces],
        token_hash_encoding='flattened chronological generated frames; little-endian signed int64 C-order',
        token_hash_shape=[horizon, width],
        unique_trajectories=len({tuple(c.flatten().tolist()) for c in codes}),
        mean_pairwise_token_difference_fraction=sum(pair) / len(pair) if pair else 0.,
        pairwise_token_difference_fraction=pair,
        different_token_positions=sum(different_per_frame),
        different_positions_per_frame=different_per_frame,
        model_log_scores=raw.tolist(), search_scores=raw.tolist(),
        score_is_behavior_logprob=False, on_policy_grpo=False,
        uses_ground_truth=False, uses_vlm=False, uses_visual_metric_reward=False,
        fixed_initial_context=True, fixed_observed_dynamic_anchor=True, recent_window=recent,
        per_frame_kv_reset=True, text_binding='provided by caller, unchanged during search',
        node_count=len(nodes), max_tree_depth=max(n.depth for n in nodes),
        nodes=[dict(node_id=n.node_id, depth=n.depth, parent=n.parent,
                    children=list(n.children), visits=n.visits,
                    value_sum=n.value_sum, mean_value=n.mean_value,
                    raw_prefix_log_score=_score(n.frames)) for n in nodes],
        simulation_traces=simulation_traces, counters=counters,
        cost_comparison='Different budget from eight-path beam; report measured counters and wall time',
        ancestry_prefix_checks=prefix_checks + horizon,
        ancestry_prefixes_equal=True, raw_logp_max_abs_error=score_error,
        frames=frame_traces,
    )
