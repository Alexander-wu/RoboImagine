"""CPU tests: causal prefix/model scores, real UCT revisits/backups, deterministic costs.

Run with a futureworlds source tree installed, or from this workspace using:
  python3 work/roboimagine_mcts_rt1_20261025/test_mcts_search.py
No CUDA, checkpoints, datasets or remote side effects are required.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
# Remote deployment ships a sealed source copy beside this test.
SOURCE_ROOT = ROOT / 'code/src'
if not SOURCE_ROOT.exists():
    SOURCE_ROOT = WORKSPACE / 'FutureWorlds/src'
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))
from futureworlds.memory import make_prefix, frame_logp
from mcts_search import rollout_mcts, _Node, _choose_child

torch.set_num_threads(1)


class CausalMock(torch.nn.Module):
    """Cumulative token+position features make cache and full scoring identical."""
    def __init__(self, vocab=13, deterministic=False):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-.4, .4, vocab))
        self.vocab = vocab
        self.deterministic = deterministic
        self.calls = []

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        if use_cache:
            assert not torch.is_grad_enabled()
            assert not self.training
        ids = input_ids if past_key_values is None else torch.cat([past_key_values, input_ids], dim=1)
        pos = torch.arange(1, ids.shape[1] + 1, device=ids.device)[None]
        key = (ids * (pos % 5 + 1)).cumsum(1).remainder(19).float()
        tokens = torch.arange(self.vocab, device=ids.device).float()[None, None]
        logits = torch.cos(key[:, :, None] * .31 + tokens * .7) + self.bias
        if self.deterministic:
            # Finite, float32-underflowed alternatives produce legal duplicates.
            logits = torch.full_like(logits, -1000.)
            logits[:, :, 0] = 0.
        logits = logits[:, -input_ids.shape[1]:]
        self.calls.append(dict(batch=input_ids.shape[0], length=input_ids.shape[1],
                               output_rows=logits.shape[0] * logits.shape[1], cache=past_key_values is not None))
        return SimpleNamespace(logits=logits, past_key_values=ids.clone() if use_cache else None)


def inputs(horizon=10, width=3):
    return (torch.tensor([[7, 8, 9, 7]]), torch.tensor([[1, 2, 0]]),
            torch.arange((horizon + 1) * 2).reshape(horizon + 1, 2).remainder(5) + 8)


def run(seed=313, horizon=10, width=3, simulations=16, deterministic=False, returned=4):
    model = CausalMock(deterministic=deterministic)
    context, anchor, actions = inputs(horizon, width)
    before = [t.clone() for t in [context, anchor, actions]]
    frames, trace = rollout_mcts(model, context, anchor, actions, horizon,
                                torch.Generator().manual_seed(seed), simulations=simulations,
                                branching=2, returned=returned, visual=7, width=width)
    for a, b in zip(before, [context, anchor, actions]):
        assert torch.equal(a, b), 'Input condition modified'
    return model, (context, anchor, actions), frames, trace


class MCTSTest(unittest.TestCase):
    def test_real_uct_revisits_and_backups(self):
        _, _, _, tr = run()
        nodes = tr['nodes']
        sims = tr['simulation_traces']
        self.assertEqual(nodes[0]['visits'], 16)
        self.assertEqual(len(sims), 16)
        self.assertEqual(tr['node_count'], 17)
        self.assertGreater(tr['max_tree_depth'], 1)
        self.assertEqual(len(nodes[0]['children']), 2)
        self.assertTrue(all(nodes[i]['visits'] > 1 for i in nodes[0]['children']))
        self.assertTrue(any(s['selections'] for s in sims[2:]))
        for node in nodes:
            seen = [s for s in sims if node['node_id'] in s['selected_tree_path']]
            self.assertEqual(node['visits'], len(seen))
            total = sum(s['terminal_value'] for s in seen)
            self.assertAlmostEqual(node['value_sum'], total, places=12)
            self.assertAlmostEqual(node['mean_value'], total / len(seen), places=12)
        for sim in sims:
            self.assertEqual(sim['backed_up_nodes'], sim['selected_tree_path'])
            self.assertAlmostEqual(sim['terminal_value'], math.exp(sim['mean_model_token_logp']), places=12)
            for selection in sim['selections']:
                choices = selection['alternatives']
                maximum = max(choices, key=lambda x: x['uct'])
                self.assertEqual(selection['selected_node'], maximum['node_id'])
                lo, hi = selection['value_min'], selection['value_max']
                for alt in choices:
                    q = .5 if hi - lo <= 1e-15 else (alt['mean_value'] - lo) / (hi - lo)
                    self.assertAlmostEqual(alt['exploitation'], min(1., max(0., q)), places=12)
                    expected_u = math.sqrt(math.log(selection['parent_visits'] + 1.) / alt['visits'])
                    self.assertAlmostEqual(alt['exploration'], expected_u, places=12)
        self.assertTrue(any(abs(a['exploitation'] - b['exploitation']) > .01
                            for s in sims for c in s['selections']
                            for a, b in [c['alternatives'][:2]]))

    def test_full_teacher_forcing_matches_cached_scores_and_rolling_history(self):
        model, (ctx, anchor, actions), frames, tr = run()
        hist = {1: anchor.expand(4, -1), **{f['target']: f['response'] for f in frames}}
        model.eval()
        for f in frames:
            self.assertEqual(tuple(f['response'].shape), (4, 3))
            self.assertEqual(tuple(f['model_logp'].shape), (4, 3))
            self.assertFalse(f['model_logp'].requires_grad)
            rebuilt, kept = make_prefix(ctx.expand(4, -1), hist, actions, f['target'])
            self.assertTrue(torch.equal(rebuilt, f['prefix']))
            with torch.no_grad():
                lp = frame_logp(model, f['prefix'], f['response'], visual=7)
            self.assertTrue(torch.allclose(lp, f['model_logp'], atol=1e-6, rtol=0))
            self.assertTrue(all(k < f['target'] for k in kept))
        self.assertEqual(tr['frames'][7]['target'], 9)
        self.assertEqual(tr['frames'][7]['retained'], [1, 3, 4, 5, 6, 7, 8])
        self.assertEqual(tr['frames'][-1]['retained'], [1, 5, 6, 7, 8, 9, 10])
        scores = sum(f['model_logp'].double().sum(-1) for f in frames)
        self.assertTrue(torch.allclose(scores, torch.tensor(tr['model_log_scores'], dtype=torch.float64), atol=1e-7, rtol=0))
        self.assertFalse(tr['score_is_behavior_logprob'])
        self.assertFalse(tr['on_policy_grpo'])

    def test_reproducibility_cost_counters_and_topk(self):
        a, _, af, at = run(seed=100)
        _, _, bf, bt = run(seed=100)
        self.assertEqual(at, bt)
        for aa, bb in zip(af, bf):
            for key in ['prefix', 'response', 'model_logp']:
                self.assertTrue(torch.equal(aa[key], bb[key]))
        expected = sorted(range(16), key=lambda i: -at['simulation_traces'][i]['raw_model_log_score'])[:4]
        self.assertEqual(at['selected_simulation_indices'], expected)
        c = at['counters']
        self.assertEqual(c['model_forward_calls'], len(a.calls))
        self.assertEqual(c['actor_token_logit_rows'], sum(x['batch'] for x in a.calls))
        self.assertEqual(c['input_token_rows'], sum(x['batch'] * x['length'] for x in a.calls))
        self.assertEqual(c['produced_logit_rows'], sum(x['output_rows'] for x in a.calls))
        self.assertEqual(c['model_forward_calls'], c['generated_frames'] * 3)
        self.assertEqual(c['expansion_frames'], 16)
        self.assertEqual(c['generated_frames'], c['expansion_frames'] + c['rollout_frames'])
        self.assertEqual(sum(not x['cache'] for x in a.calls), c['generated_frames'])
        self.assertGreater(c['generated_token_rows'], 8 * 10 * 3)
        self.assertEqual(at['ancestry_prefix_checks'], 16 * 10 + 10)
        json.dumps(at, allow_nan=False)

    def test_duplicate_trajectories_and_terminal_revisits_are_preserved(self):
        _, _, frames, tr = run(horizon=1, deterministic=True)
        self.assertEqual(tr['node_count'], 3)
        self.assertEqual(tr['counters']['generated_frames'], 2)
        self.assertEqual(sum(s['terminal_revisit'] for s in tr['simulation_traces']), 14)
        self.assertEqual(tr['selected_simulation_indices'], [0, 1, 2, 3])
        self.assertEqual(tr['unique_trajectories'], 1)
        self.assertEqual(frames[0]['response'].shape[0], 4)
        self.assertTrue(torch.equal(frames[0]['response'][0], frames[0]['response'][1]))
        for n in tr['nodes']:
            self.assertGreater(n['visits'], 0)

    def test_every_completed_simulation_has_verifiable_token_hash(self):
        _, _, frames, trace = run()
        hashes = trace['all_simulation_token_sha256']
        self.assertEqual(len(hashes), 16)
        self.assertTrue(all(len(h) == 64 for h in hashes))
        self.assertEqual(hashes, [s['trajectory_token_sha256'] for s in trace['simulation_traces']])
        for row, sim in enumerate(trace['selected_simulation_indices']):
            values = torch.cat([f['response'][row] for f in frames]).numpy().astype('<i8', copy=False)
            self.assertEqual(hashlib.sha256(values.tobytes(order='C')).hexdigest(), hashes[sim])

    def test_returned_tensors_do_not_alias_input_or_sibling_frames(self):
        _, (ctx, anchor, actions), frames, tr = run()
        snapshots = [t.clone() for t in [ctx, anchor, actions]]
        other = frames[0]['response'][1].clone()
        later_prefix = frames[1]['prefix'].clone()
        frames[0]['response'][0].fill_(6)
        self.assertTrue(torch.equal(frames[0]['response'][1], other))
        self.assertTrue(torch.equal(frames[1]['prefix'], later_prefix))
        frames[0]['prefix'].fill_(1)
        for old, new in zip(snapshots, [ctx, anchor, actions]):
            self.assertTrue(torch.equal(old, new))
        self.assertEqual(tr['raw_logp_max_abs_error'], 0.)

    def test_real_llama_dynamic_cache_and_text_conditioning(self):
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            self.skipTest('transformers unavailable; mock causal tests still mandatory')
        from futureworlds.text_conditioning import TextWorldModel
        torch.manual_seed(29)
        core = LlamaForCausalLM(LlamaConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=20,
            max_position_embeddings=128, attention_dropout=0.,
        ))
        model = TextWorldModel(core, text_width=8, layers=(0,), inner=8).eval()
        with torch.no_grad():
            model.adapters['0'].out.weight.normal_(std=.03)
        features = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.long)
        model.set_text(features, mask)
        ctx, anchor, actions = inputs(8)
        frames, trace = rollout_mcts(model, ctx, anchor, actions, 8,
                                    torch.Generator().manual_seed(91), visual=7, width=3)
        self.assertIs(model._text, features)
        self.assertIs(model._mask, mask)
        for frame in frames:
            with torch.no_grad():
                actual = frame_logp(model, frame['prefix'], frame['response'], visual=7)
            self.assertTrue(torch.allclose(actual, frame['model_logp'], atol=2e-6, rtol=0))
        self.assertEqual(trace['frames'][-1]['retained'], [1, 3, 4, 5, 6, 7, 8])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_uct_unvisited_and_tie_breaks(self):
        nodes = [_Node(0, 0, None, [], children=[1, 2], visits=2),
                 _Node(1, 1, 0, [], visits=2, value_sum=.8),
                 _Node(2, 1, 0, [])]
        chosen, trace = _choose_child(nodes[0], nodes, .2, .5, 1.)
        self.assertEqual(chosen, 2)
        self.assertTrue(trace['unvisited_first'])
        nodes[2].visits, nodes[2].value_sum = 2, .8
        chosen, trace = _choose_child(nodes[0], nodes, .4, .4, 1.)
        self.assertEqual(chosen, 1)
        self.assertTrue(all(c['exploitation'] == .5 for c in trace['alternatives']))

    def test_future_actions_cannot_change_first_frame(self):
        # Later actions affect later frame prompts, but never the first transition.
        ctx, anchor, actions = inputs(1)
        model = CausalMock()
        fa, _ = rollout_mcts(model, ctx, anchor, actions, 1, torch.Generator().manual_seed(8),
                             visual=7, width=3, simulations=2, returned=2)
        actions[0].fill_(12)  # unused action before the permanent anchor
        fb, _ = rollout_mcts(model, ctx, anchor, actions, 1, torch.Generator().manual_seed(8),
                             visual=7, width=3, simulations=2, returned=2)
        self.assertTrue(torch.equal(fa[0]['response'], fb[0]['response']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
