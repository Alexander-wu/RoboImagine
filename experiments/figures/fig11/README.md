# Figure 11 — Prediction distances and feedback

[English](README.md) | [简体中文](README.zh-CN.md)

## Paper correspondence and protocol

This uses all 128 matched RT-1 reporting trajectories and compares final predictions of SFT, MCTS, ordinary beam and GDBS. Distances are between models, not between candidates inside a single search. Higher token distance alone does not mean higher quality. Semantic feedback comes from the frozen training Jev evaluator; it is distinct from the independent Qwen ratings in Table 3. Both feedback columns are higher-is-better.

## Reproduce

Install dependencies using the [experiment guide](../../README.md). From the repository root:

```bash
python experiments/run_all.py --figures 11
```

Alternatively, open [Reproduce.ipynb](Reproduce.ipynb) and Run All in the environment with the dependencies installed.

Generated PDF: `experiments/reproduced/figures/fig11/figure.pdf`. The same directory contains the run log.

## Inputs and code

- [results.json](results.json)
- [per_case.csv](per_case.csv)
- [distance_pairs.csv](distance_pairs.csv)
- [quality_reference.json](quality_reference.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

This entry redraws archived outputs; it does not retrain, regenerate the evaluation cohort, or call paid evaluators. Use the [inference guide](../../../docs/INFERENCE.md) to generate new predictions from public weights.
