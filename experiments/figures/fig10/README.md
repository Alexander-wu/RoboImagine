# Figure 10 — Three-dataset SFT comparison

**Actual experiment computation:** [rt1_visual](../../executed_sources/rt1_visual/README.md) · [dataset_suite](../../executed_sources/dataset_suite/README.md)
This page’s notebook redraws saved results; the links above lead to the original training, generation and scoring programs.

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

The RT-1 example uses the visual-only 500-update variant. BridgeV2 and RoboCasa use combined-feedback 500-update variants. In particular, the archived Bridge example predates the current validation-selected 250-update public checkpoint; it is not interchangeable with Figure 6 or the current main-table checkpoint. The selection manifest identifies the assets.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 10
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig10/r2_three_dataset_comparison.pdf`. The same directory contains the run log.

## Inputs and code

- [selection.json](selection.json)
- [render_figure.py](render_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
