# Executed experiment sources

[简体中文](README.zh-CN.md) | [Project](../../README.md) | [Archived figure redraws](../README.md)

This directory contains the **actual training, inference, metric, reward and aggregation programs used on the server**, read from the completed experiments. It does not turn paper table entries into “new results.” No training, prediction, judge call or metric recomputation was performed to prepare this source release.

## Start with the computation

- **Visual metrics from predicted pixels and GT:** [original `Metrics` and evaluation loop](rt1_hybrid/code/src/futureworlds/evaluation.py). Per-frame PSNR/SSIM/VGG-LPIPS/MAE are computed from float predictions, followed by frame averages and trajectory averages in the experiment worker.
- **Candidate generation:** [GDBS](rt1_hybrid/code/src/futureworlds/search.py), [ordinary beam](search_ablation/code/src/futureworlds/ordinary_search.py), [sampling / rolling history](search_ablation/code/src/futureworlds/memory.py), [actual MCTS implementation](mcts/source_v1/mcts_search.py).
- **Post-training:** [original policy optimization](rt1_hybrid/code/src/futureworlds/training.py), [candidate/feedback/update loop](rt1_hybrid/train.py), [visual reward and feedback combination](rt1_hybrid/common.py), [pairwise judge service](rt1_hybrid/judge_service.py).
- **Continued SFT:** [final four-GPU training](continued_sft/source_sft_v4/sft/train.py), [effective batch and initialization](continued_sft/source_sft_v4/sft/sft_common.py). The eight-GPU `protocol.json` records the earlier cache preparation; the admission record and v4 code define the completed four-GPU continuation.
- **Independent semantic evaluation:** [Qwen prompt and parser](qwen97/source/original_common.py), [video inference and format-repair attempts](qwen97/source/worker.py), [paired aggregation](qwen97/source/aggregate.py).
- **Window cost:** [profiled decoding](window30/source/profile_decoder.py), [generation and pixel scoring](window30/source/worker.py), [complete-grid validation, repeated timings and bootstrap](window30/source/aggregate.py).

## Paper-to-source map

| Paper | Experiment | Actual code | Configuration |
|---|---|---|---|
| Table 2/3/4; Figures 8/9/10 | [RT-1 Hybrid / GDBS](rt1_hybrid/README.md) | [Execution](rt1_hybrid/train.py) · [Detail / scoring](rt1_hybrid/common.py) | [Protocol](rt1_hybrid/protocol.json) |
| Table 3(A); Figure 10 | [RT-1 visual-only](rt1_visual/README.md) | [Execution](rt1_visual/train.py) · [Detail / scoring](rt1_visual/common.py) | [Protocol](rt1_visual/protocol.json) |
| Table 3(A) | [RT-1 semantic-only](rt1_semantic/README.md) | [Execution](rt1_semantic/train.py) · [Detail / scoring](rt1_semantic/judge_service.py) | [Protocol](rt1_semantic/protocol.json) |
| Table 3(B)/4; Figure 8 | [Sampling / ordinary beam](search_ablation/README.md) | [Execution](search_ablation/train.py) · [Detail / scoring](search_ablation/common.py) | [Protocol](search_ablation/search_ordinary/protocol.json) |
| Table 2; Figure 10 | [RoboCasa / earlier Bridge suite](dataset_suite/README.md) | [Execution](dataset_suite/train.py) · [Detail / scoring](dataset_suite/evaluate.py) | [Protocol](dataset_suite/robocasa_hybrid500/protocol.json) |
| Table 2; Figure 6 | [Bridge selected checkpoint](bridge_selected/README.md) | [Execution](bridge_selected/source_v1/train_retry.py) · [Detail / scoring](bridge_selected/source_v1/evaluate_retry.py) | [Protocol](bridge_selected/configs/lr2e7/protocol.json) |
| Table 4; Figures 8/11 | [Likelihood-guided MCTS](mcts/README.md) | [Execution](mcts/source_v1/train_mcts.py) · [Detail / scoring](mcts/source_v1/mcts_search.py) | [Protocol](mcts/config_v1/protocol.json) |
| Figures 8/9 | [Continued SFT](continued_sft/README.md) | [Execution](continued_sft/source_sft_v4/sft/train.py) · [Detail / scoring](continued_sft/source_sft_v4/sft/evaluate.py) | [Protocol](continued_sft/sft/rt1_sft500_v1/four_gpu_preparation_admission.json) |
| Figure 8 | [SFT / Hybrid intermediate checkpoints](learning_curves/README.md) | [Execution](learning_curves/source_v1/evaluator.py) · [Detail / scoring](learning_curves/source_v1/evaluator.py) | [Protocol](learning_curves/hybrid_000250/protocol.json) |
| Figure 8 | [MCTS / ordinary intermediate checkpoints](search_curves/README.md) | [Execution](search_curves/v1/source/worker.py) · [Detail / scoring](search_curves/v1/source/worker.py) | [Protocol](search_curves/v1/rank0/protocol.json) |
| Table 3; Figure 7 (Qwen) | [Independent six-dimensional scoring](qwen97/README.md) | [Execution](qwen97/source/worker.py) · [Detail / scoring](qwen97/source/aggregate.py) | [Protocol](qwen97/source/cohort.json) |
| Figure 11 | [Between-model feedback](prediction_feedback/README.md) | [Execution](prediction_feedback/score_controller_v1.py) · [Detail / scoring](prediction_feedback/scoring_v1/judge_service_h32.py) | [Protocol](prediction_feedback/scoring_v1/protocol.json) |
| Figure 13 | [Historical long-horizon memory](long_memory/README.md) | [Execution](long_memory/code_v2/worker.py) · [Detail / scoring](long_memory/code_v2/aggregate.py) | [Protocol](long_memory/eval_v2/protocol.json) |
| Table 5; Figure 14 | [Matched-30 window / efficiency](window30/README.md) | [Execution](window30/source/worker.py) · [Detail / scoring](window30/source/aggregate.py) | [Protocol](window30/protocol.json) |

Figure 12's saved search trace and selected frames remain in its [case archive](../figures/fig12/README.md); the GDBS algorithm producing the ancestry is linked above. Table 1's author-provided RoboAlign-Judge aggregate row and Figure 7's Doubao/session-model summaries do not acquire raw scoring code by being exported: missing original evaluators are not fabricated here.

## Inputs and execution order

1. **Prepare inputs.** [Original RT-1 preparation](rt1_semantic/prepare.py) and [Bridge/RoboCasa preparation](dataset_suite/prepare.py) read the actual images/actions/instruction and cache initial context, dynamic anchor, actions and text features. Training selection has 5,200 windows, 8 smoke windows and 128 reporting windows in these runs. Keep the actual IDs/splits from each protocol; historical `dev` directory labels do not authorize relabeling validation as test.
2. **Load initialization and train only when reproducing training.** `common.py` loads the codec, SFT predictor, text adapters and frozen T5; `train.py` generates candidates, computes feedback, updates the actor and writes checkpoints/receipts. The frozen reference starts at the same initializer. Training is not needed to score an already saved prediction.
3. **Generate predictions from the chosen checkpoint.** The per-experiment evaluation worker loads that checkpoint, predicts from the observed prefix/actions/instruction, and saves float pixels and tokens. Evaluation beam width 4 is distinct from candidate search during post-training.
4. **Score saved predictions against GT.** Use the original `Metrics` implementation above; do not score a rendered JPEG or read a paper mean as input. Semantic workers receive videos and task context, save actual model responses, and parse them using the archived prompt/schema.
5. **Aggregate completed records.** Original aggregators check identities and grids before averaging. The fixed-30 experiment has 750 quality records and 2,250 timing records, not 90 independent trajectories. It is distinct from the 128-trajectory main evaluation and the historical 11-trajectory long-horizon subset.
6. **Draw figures last.** The existing figure notebooks consume saved records/summaries. They are visualization utilities, not experimental inference or training.

### Existing public checkpoint inference

For a new prediction and actual pixel-based scores on a released demonstration input, the previously released interface is:

```bash
python -m pip install -e '.[evaluation,notebook]'
roboimagine --dataset rt1 --variant main --example-index 0 --metrics --output predictions/rt1_main
roboimagine --dataset rt1 --variant sft --example-index 0 --metrics --output predictions/rt1_sft
```

These commands load weights, generate pixels and evaluate against the input GT. They do not recreate an entire paper cohort. See [inference instructions](../../docs/INFERENCE.md) and the [existing inference notebook](../../notebooks/Inference.ipynb). No command above was run as part of this source-organization task.

### Historical worker entry points

The following are **entry-point references after restoring their input layout, dependencies, protocol and provenance gates**. They are not copy-and-run full-cohort recipes with this checkout alone. Do not invoke an old scheduler against a live cluster.

| Experiment | Existing invocation form | Required context |
|---|---|---|
| RT-1 Hybrid | `torchrun --standalone --nproc_per_node=8 worker_entry.py train` | In `rt1_hybrid`; all 8 CUDA devices visible; its prepared cache and gate |
| Sampling / ordinary | `torchrun --standalone --nproc_per_node=8 worker_entry.py train` | In `search_ablation`; `R2_RUN_DIR` chooses `search_sample` or `search_ordinary` |
| MCTS | `torchrun --standalone --nproc_per_node=4 worker_entry.py train_mcts` | In `mcts/source_v1`; `R2_RUN_DIR` + fresh `R2_OUTPUT_ROOT`; checkpoint receipts for `--resume` |
| Continued SFT | `torchrun --standalone --nproc_per_node=4 worker_entry.py train` | In `continued_sft/source_sft_v4/sft`; `R2_SFT_OUTPUT`, cache preparation and admission |
| Bridge evaluation | `torchrun --standalone --nproc_per_node=4 worker_entry.py evaluate_retry` | In `bridge_selected/source_v1`; `R2_RUN_DIR`, `BRIDGE_EVAL_SPEC` with cohort/checkpoint file hashes/output |
| Qwen | `python worker.py` then `python aggregate.py` after all workers finish | In `qwen97/source`; one GPU per worker, unique `EVALUATION_RANK`, prepared queue and pinned local judge |
| Window30 | `python worker.py --rank 0`, then `python worker.py --rank 0 --score` | In `window30/source`; ranks 0–3, one GPU each, fixed prepared inputs/protocol; aggregate only after all ranks |
| Historical long memory | `python worker.py --rank 0 --shards 4` | In `long_memory/code_v2`; ranks 0–3 and original separate cohort manifests |

The archive retains the `futureworlds` Python namespace because those are the executed imports. This is an internal source namespace, not a new claim about the manuscript's positioning. Each experiment retains its own runtime snapshot where available; do not silently substitute a different snapshot merely because filenames match.

## Portability, provenance and remaining assets

Machine filesystem roots are replaced by `/opt/roboimagine-workspace` and `/opt/roboimagine-scratch`; device UUIDs use `GPU-REPLACE-UUID-*`. These are placeholders, not public download locations. Algorithm statements, numerical hyperparameters, sampling/averaging logic and score prompts are unchanged. An AST comparison during packaging checks that Python changes consist only of these string substitutions.

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) contains both original and published SHA256 hashes for every file. **Embedded source/input/checkpoint hash gates still describe the historical run.** They can reject relocated sources or missing receipts; they are not disabled or rewritten to make an unverified run appear verified. Porting a full run requires making a new local protocol/manifest recording the relocation and available asset hashes. Preserve the original record as provenance.

The existing [HF release](https://huggingface.co/easylearning/RoboImagine) has main/SFT weights and demonstration inputs. Full prepared training/reporting cohorts, all intermediate/ablation checkpoints, original cache/receipt trees and the evaluator models are not all in that release. A source export is therefore **not a claim that all paper experiments run from a fresh clone**. No guessed public asset URLs or synthetic result files are supplied. Bridge's canceled second learning-rate arm and unrelated cluster schedulers are excluded.

Use Python 3.11+ for the historical sources (`hashlib.file_digest` is used), a compatible CUDA/PyTorch environment, the package dependencies in [pyproject.toml](../../pyproject.toml), and the original LPIPS/PIQA metric dependencies. Frozen judge integrations must use their recorded versions and local files; see the protocol and service code. A complete historical environment lock is not asserted.

Verify this source export without launching any experiment:

```bash
python experiments/executed_sources/verify_snapshot.py
```

This checks source hashes and parses Python only; it does not reproduce model results. The export intentionally excludes model-architecture illustration code (Figures 1–5).
