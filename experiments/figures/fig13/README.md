# Figure 13 — Long-horizon memory comparison

**Actual experiment computation:** [long_memory](../../executed_sources/long_memory/README.md)
This page’s notebook redraws saved results; the links above lead to the original training, generation and scoring programs.

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

The plotted historical experiment uses the same 11 held-out trajectories for the first 64 and all 128 predicted frames. Full memory means a fixed dynamic anchor plus six recent states; current-frame-only removes the dynamic anchor and retains one recent state. Both retain initial context. This comparison does not isolate the anchor contribution. The data file also retains a historical development cohort; the figure selects the 11-trajectory held-out cohort. Do not relabel this as the newer matched-30 experiment.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 13
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig13/figure.pdf`. The same directory contains the run log.

## Inputs and code

- [data.json](data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
