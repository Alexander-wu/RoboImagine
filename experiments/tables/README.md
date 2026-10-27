# Paper tables

**Computation provenance:** [Actual training/inference/scoring/aggregation sources](../executed_sources/README.md) map Tables 2–5 to the executed code. Table renderers only typeset saved results; Table 1’s author-provided aggregates are not newly evaluated scores.

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment guide](../README.md)

## Run

From the repository root, after installing the experiment dependencies:

```bash
python experiments/tables/build_tables.py
```

The [builder](build_tables.py) compiles five standalone preview pages into `experiments/reproduced/tables/tables.pdf`, retaining an execution receipt and build log. TeX and `latexmk` are required; a missing compiler is recorded as a skip. The [Run All notebook](../Reproduce_All.ipynb) invokes this builder after the figure replay.

## Table correspondence

| Table | Source | Interpretation |
|---|---|---|
| [1](table01.tex) | [Main-table CSV](data/our_model_main_tables.csv) | Author-provided aggregate RoboAlign-Judge means and deviations; not reconstructed from raw judge responses |
| [2](table02.tex) | [Main-table CSV](data/our_model_main_tables.csv) | Our three-dataset visual results; RT-1/RoboCasa reporting cohorts have 128 trajectories; Bridge uses 32 validation trajectories and a validation-selected checkpoint |
| [3](table03.tex) | [Visual](data/table03_visual_results.json), [Qwen](data/table03_qwen97_results.json) | Feedback/search ablations: 128 trajectories for visual metrics, the same 97 paired valid trajectories across all six Qwen arms |
| [4](table04.tex) | [Search results](data/table04_search_results.json) | Final 500-update sampling/MCTS/ordinary-beam/GDBS models versus SFT on all 128 trajectories, with training GPU-seconds per trajectory presentation |
| [5](table05.tex) | [Matched-30 results](data/table05_matched30_results.json) | Fixed-checkpoint anchored windows W=1/3/6/12 over the same 30 test trajectories at H=16/32/48/64/96 |

Tables 1–2 include only our model rows and omit external-baseline ranks. Tables 3–5 retain the relevant controls. The checked-in TeX contains the paper's formatted values; JSON/CSV files preserve source summaries and protocol. This command recompiles the presentation, not new training, inference or VLM scoring.

## Details needed to interpret the ablations

- Table 3(A) Hybrid and Table 3(B) GDBS denote the **same configuration**, with identical visual metrics and current paired Qwen scores. The old nine-case search pilot is no longer the current table. Three sampling responses use the bounded format-repair records preserved in the Qwen JSON.
- Independent Qwen evaluation is separate from the Jev training feedback. Different metrics need not rank models identically.
- Table 4 cost is allocated GPU count × training-loop wall time / (updates × global batch). It is not per-unique-video inference latency, nor proof of equal compute budgets. The MCTS configuration is specified in the archived protocol.
- Table 5 quality averages frames, then trajectories. Three timing repeats are averaged within each trajectory before averaging over 30 trajectories. The source retains 750 quality records and 2,250 timing records including current-frame-only; the displayed table focuses on four anchored windows. The source also includes paired-bootstrap summaries under one checkpoint.
- W6/H32 PSNR is 22.20 dB on this long-eligible cohort, while the original 128-trajectory main result is 22.57 dB. These are different cohort means. Figure 13 is the separate historical 11-trajectory experiment.

See the [full ablation protocols](../README.md#ablations-what-changes-and-what-stays-fixed) and [model inference guide](../../docs/INFERENCE.md) for what can currently be rerun from weights.
