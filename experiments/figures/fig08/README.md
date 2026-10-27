# Figure 8 — Continued-SFT and search-variant learning curves

**Actual experiment computation:** [continued_sft](../../executed_sources/continued_sft/README.md) · [learning_curves](../../executed_sources/learning_curves/README.md) · [search_curves](../../executed_sources/search_curves/README.md)
This page’s notebook redraws saved results; the links above lead to the original training, generation and scoring programs.

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

Continued SFT, GDBS, ordinary beam and MCTS are evaluated at 100/250/500 updates on the same 128 RT-1 reporting trajectories with 32-frame predictions and a shared evaluation decoder. The qualitative row uses drawer case034 at predicted frame 32. Equal updates do not imply equal compute. Intermediate checkpoints are not part of the released main/SFT checkpoint bundle.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 8
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig08/figure.pdf`. The same directory contains the run log.

## Inputs and code

- [curve_data.json](curve_data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
