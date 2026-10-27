# Archived experiment figures and tables

[English](README.md) | [简体中文](README.zh-CN.md) | [Project home](../README.md)

## Actual experiments versus redrawing

For training, prediction, scoring and aggregation, use the [actual executed server sources](executed_sources/README.md), with a [paper-to-source map](executed_sources/README.md#paper-to-source-map). The [public inference guide](../docs/INFERENCE.md) generates new predictions from main/SFT weights and supports pixel-based metrics.

This directory's figure/table notebooks **redraw saved outputs**. Running them is not training, model evaluation, or verification of the paper's numerical claims. Original MCTS, feedback, continued-SFT and window worker code is now separately archived under `executed_sources/`; full datasets/ablation checkpoints and relocation of source gates remain requirements for a full run.

## Installation and Run All

From the repository root, use a dedicated Python 3.10+ environment:

```bash
python -m venv .venv-plots
source .venv-plots/bin/activate
python -m pip install -r experiments/requirements.txt
cd experiments
jupyter lab Reproduce_All.ipynb
```

Select this environment's kernel and choose **Run All**. The [notebook](Reproduce_All.ipynb) redraws Figures 6–14 and attempts Table 1–5 previews. Equivalent commands, from `experiments/`:

```bash
python run_all.py
python tables/build_tables.py
# A smaller selection:
python run_all.py --figures 8 11 14
```

CairoSVG requires system Cairo (`brew install cairo` on macOS, or the `libcairo2` package on Linux). `CAIRO_LIBRARY` can identify the library if discovery fails. Table PDFs additionally require TeX and `latexmk` with the packages referenced by [table_style.tex](tables/table_style.tex). Without TeX, the table builder records a skip; that is not a successful table compilation. Figures can still be reproduced. [requirements-tested.txt](requirements-tested.txt) records the earlier local rendering environment, rather than promising bitwise rendering parity on every platform.

Outputs are written under `experiments/reproduced/`: figure PDFs and logs in `figures/figNN/`, table previews in `tables/tables.pdf`, and execution receipts. [input_checksums.json](input_checksums.json) is checked before and after figure replay. Source data are not overwritten. Fonts or rendering-library changes can affect appearance without changing the measurements. No plot notebook submits Ray jobs, trains models or makes paid API requests.

## Paper figure index

| Paper figure | Content | Guide | Notebook | Code |
|---|---|---|---|---|
| 6 | Four-timestamp prediction comparison | [EN](figures/fig06/README.md) / [中文](figures/fig06/README.zh-CN.md) | [Run All](figures/fig06/Reproduce.ipynb) | [build_figure.py](figures/fig06/build_figure.py) |
| 7 | Multi-model evaluation and excerpts | [EN](figures/fig07/README.md) / [中文](figures/fig07/README.zh-CN.md) | [Run All](figures/fig07/Reproduce.ipynb) | [build_figure.py](figures/fig07/build_figure.py) |
| 8 | Continued-SFT and search-variant learning curves | [EN](figures/fig08/README.md) / [中文](figures/fig08/README.zh-CN.md) | [Run All](figures/fig08/Reproduce.ipynb) | [build_figure.py](figures/fig08/build_figure.py) |
| 9 | Training cost and prediction quality | [EN](figures/fig09/README.md) / [中文](figures/fig09/README.zh-CN.md) | [Run All](figures/fig09/Reproduce.ipynb) | [build_figure.py](figures/fig09/build_figure.py) |
| 10 | Three-dataset SFT comparison | [EN](figures/fig10/README.md) / [中文](figures/fig10/README.zh-CN.md) | [Run All](figures/fig10/Reproduce.ipynb) | [render_figure.py](figures/fig10/render_figure.py) |
| 11 | Prediction distances and feedback | [EN](figures/fig11/README.md) / [中文](figures/fig11/README.zh-CN.md) | [Run All](figures/fig11/Reproduce.ipynb) | [build_figure.py](figures/fig11/build_figure.py) |
| 12 | GDBS search case study | [EN](figures/fig12/README.md) / [中文](figures/fig12/README.zh-CN.md) | [Run All](figures/fig12/Reproduce.ipynb) | [build_figure.py](figures/fig12/build_figure.py) |
| 13 | Long-horizon memory comparison | [EN](figures/fig13/README.md) / [中文](figures/fig13/README.zh-CN.md) | [Run All](figures/fig13/Reproduce.ipynb) | [build_figure.py](figures/fig13/build_figure.py) |
| 14 | Window sensitivity and inference efficiency | [EN](figures/fig14/README.md) / [中文](figures/fig14/README.zh-CN.md) | [Run All](figures/fig14/Reproduce.ipynb) | [build_figure.py](figures/fig14/build_figure.py) |

## Paper table index

| Table | Experiment | Source / guide |
|---|---|---|
| 1 | Six-dimensional RoboAlign-Judge aggregate scores | [Our main-table rows](tables/data/our_model_main_tables.csv) · [guide](tables/README.md) |
| 2 | Three-dataset visual prediction quality | [Our main-table rows](tables/data/our_model_main_tables.csv) · [Table 2](tables/table02.tex) |
| 3 | Feedback composition and training candidate search | [Visual results](tables/data/table03_visual_results.json) · [Qwen results](tables/data/table03_qwen97_results.json) |
| 4 | Full search comparison and training cost | [Results and protocol](tables/data/table04_search_results.json) · [Table 4](tables/table04.tex) |
| 5 | History-window sensitivity and efficiency | [Matched-30 results](tables/data/table05_matched30_results.json) · [Figure 14](figures/fig14/README.md) |

See the [English table guide](tables/README.md) / [中文表格教程](tables/README.zh-CN.md) for compilation and provenance. Table 1 contains author-provided aggregate means/deviations; this package does not reconstruct those deviations from raw judge outputs. Table 2's RT-1/RoboCasa rows use the 128-trajectory reporting cohorts, while Bridge uses 32 validation trajectories and a validation-selected checkpoint. The released demo clips are not substitutes for these cohorts.

## Ablations: what changes and what stays fixed?

### Feedback composition — Table 3(A)

Compare SFT initialization, semantic-only (Jev), visual-only, and combined (Hybrid) feedback. The post-trained variants share initialization, 500 updates, the RT-1 reporting cohort and evaluation decoding. GDBS supplies the training candidates. Visual metrics average 32 predicted frames over 128 trajectories. This isolates feedback composition within the recorded runs; it is not a multi-seed result.

**Hybrid in block A and GDBS in block B are the same configuration.** Their visual results and current paired Qwen score must agree. The current Qwen file evaluates all six arms on the **same 97 valid paired trajectories**; the older nine-trajectory search pilot is superseded. Three malformed sampling responses receive bounded format-repair retries, recorded alongside the original protocol in the source JSON. This independent Qwen3-VL-8B-Thinking evaluator is different from the frozen Jev model used for training feedback.

### Candidate search — Table 3(B), Table 4, Figures 8/11/12

Random sampling, ordinary beam and GDBS change how training futures are proposed while retaining combined feedback. GDBS uses four groups with two beams per group and returns one trajectory per group; ordinary beam uses a global pool. Table 4 additionally evaluates the implemented likelihood-guided MCTS variant and reports the final 500-update models on all 128 trajectories, replacing the early eight-trajectory pilot.

The MCTS protocol uses frame-prefix UCT, 16 rollouts, branching factor 2 and exploration coefficient 1; it returns four trajectories by raw log probability. No ground-truth or VLM reward guides its tree search. Feedback is applied to selected candidates for policy training. This specific MCTS configuration is not a claim about every MCTS algorithm, and its compute budget differs from beam search. Exact archived settings are in [table04_search_results.json](tables/data/table04_search_results.json).

Figure 11 measures **between-model final-prediction** token distances and training-evaluator feedback. Figure 12 instead examines **within-search retained ancestry** for a selected training case. Neither token distance nor a selected example is itself a population-level quality improvement.

### Extra training versus post-training — Figures 8/9

Continued SFT and the RL variants start from the shared SFT checkpoint. Figure 8 plots actual evaluated checkpoints at 100/250/500 updates with shared evaluation decoding. Figure 9 reports allocated GPU count × training-loop wall time, including in-loop search/feedback costs. Table 4 divides the same kind of training GPU-seconds by updates × global batch size to give cost per trajectory **presentation**, not per unique trajectory. These costs are separate from warm inference latency in Table 5/Figure 14.

### Bounded history — Table 5 and Figures 13/14

No training is performed in this sweep. The checkpoint is fixed, and the recent-history window changes. Window size excludes the fixed dynamic anchor; fixed initial context is retained. The source grid contains current-frame-only plus anchored W=1/3/6/12.

Table 5/Figure 14 use the same 30 test trajectories at H=16/32/48/64/96, drawn with seed 20261026 from 38 eligible instructed long sequences with at least 98 real frames. This gives 750 quality records and 2,250 generation-timing records. Repeated timing is averaged within each trajectory before averaging over 30 trajectories; three repeats are not 90 independent trajectories. Quality averages frames first, then trajectories. The stored summaries include 95% paired trajectory-bootstrap intervals (10,000 resamples) under one checkpoint.

Warm full-rollout time excludes model loading, text-cache construction, scoring and file I/O. Peak memory is PyTorch allocated memory including resident models; it is not total reserved VRAM. Evaluation uses beam width 4, FP32 and TF32 disabled, with GPU exclusivity checked in the experiment records.

**The W6/H32 PSNR of 22.20 dB belongs to these 30 long-eligible trajectories. The main-table 22.57 dB belongs to the original 128-trajectory cohort. Do not directly compare the two means.** Figure 13 retains a separate historical 11-trajectory held-out experiment at 64/128 frames; it is not the matched-30 sweep and does not isolate the anchor's independent contribution.

## New predictions versus archived figure replay

For public-weight predictions, use [the inference tutorial](../docs/INFERENCE.md). The bundle provides main and SFT checkpoints for three datasets and two H32 examples per dataset. Full prepared evaluation inputs, intermediate search checkpoints and reward-only/MCTS/ordinary-beam trained checkpoints are not in this bundle. Longer predictions require real actions of sufficient length; running the six examples cannot reconstruct the cohort means above.

## Data integrity and attribution

[figure_index.json](figure_index.json) maps figure numbers to scripts and output names; [input_checksums.json](input_checksums.json) pins replay inputs. [Release provenance](../provenance/) records checkpoint validation and download checks. Original source checkpoint hashes and experiment identifiers are preserved where available; private server endpoints and credentials are not required. Retained font/icon notices and [NOTICE](../NOTICE.md) apply to redistributed assets. The release does not grant blanket rights to third-party datasets or baseline implementations.
