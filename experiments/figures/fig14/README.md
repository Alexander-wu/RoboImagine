# Figure 14 — Window sensitivity and inference efficiency

**Actual experiment computation:** [window30](../../executed_sources/window30/README.md)
This page’s notebook redraws saved results; the links above lead to the original training, generation and scoring programs.

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

A fixed Hybrid500 checkpoint is evaluated on the same 30 long-sequence-eligible RT-1 test trajectories at H=16/32/48/64/96. The figure shows anchored W=1/3/6/12; source results additionally retain current-frame-only. Quality averages frames first and trajectories second. Three timing repeats are averaged within each trajectory. Warm generation time excludes model loading, text-cache construction, scoring and file I/O; peak memory is PyTorch allocated memory with resident models, not reserved memory. W6/H32 PSNR is 22.20 dB here versus 22.57 dB in the original 128-trajectory main cohort; these means are not directly comparable.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 14
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig14/figure14_heatmap_bars.pdf`. The same directory contains the run log.

## Inputs and code

- [data.json](data.json)
- [plotted_data.json](plotted_data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
