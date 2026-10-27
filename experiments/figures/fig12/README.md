# Figure 12 — GDBS search case study

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

An actual training case is selected from eight conditions by the increase in semantic-score spread. Retained ancestry and feedback come from saved traces, not a hand-drawn synthetic tree. This selected illustration is not an average over the test set. Inspect raw_traces.json and expected_ancestry.json to trace the retained branches.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 12
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig12/gdbs_case_study.pdf`. The same directory contains the run log.

## Inputs and code

- [raw_traces.json](raw_traces.json)
- [expected_ancestry.json](expected_ancestry.json)
- [asset_manifest.json](asset_manifest.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
