# Figure 9 — Training cost and prediction quality

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

GPU-hours equal allocated GPU count multiplied by training-loop wall time: 1.063 for continued SFT and 28.152 for GDBS in the archived 500-update runs. The former uses four H20s and the latter eight. The interval includes in-loop search, feedback, synchronization and checkpointing, but excludes initialization, preprocessing, final evaluation and the preceding 50k SFT updates. These are single-run training costs, not inference latency or an equal-compute comparison.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 9
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig09/figure.pdf`. The same directory contains the run log.

## Inputs and code

- [data.json](data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
