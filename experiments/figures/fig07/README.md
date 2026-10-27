# Figure 7 — Multi-model evaluation and excerpts

**Actual experiment computation:** [qwen97](../../executed_sources/qwen97/README.md)
This page’s notebook redraws saved results; the links above lead to the original training, generation and scoring programs.

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

The paired cohorts contain 97 trajectories for Qwen, 10 for Doubao and 50 for GPT-6 Astra. They are not one shared cross-evaluator cohort. GPT-6 Astra denotes the configured-model session review, not a pinned API model revision. The right-hand excerpts are individual recorded responses, separate from the aggregate bars. This replay does not request new model ratings.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 7
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig07/figure.pdf`. The same directory contains the run log.

## Inputs and code

- [data.json](data.json)
- [excerpts.json](excerpts.json)
- [session_model_provenance.json](session_model_provenance.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
