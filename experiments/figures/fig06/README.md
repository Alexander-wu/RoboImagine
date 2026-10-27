# Figure 6 — Four-timestamp prediction comparison

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

RT-1 case032 and BridgeV2 case079 use saved predictions at frames 8, 16, 24 and 32 with fixed within-case crops. Bridge uses the current validation-selected checkpoint. These are selected qualitative examples, not aggregate metrics. Necessary baseline frames are included; baseline model implementations are not.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 6
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig06/baseline_comparison.pdf`. The same directory contains the run log.

## Inputs and code

- [selection.json](selection.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
