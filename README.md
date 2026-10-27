# RoboImagine

**Learning Robotic World Models with Semantic and Visual Feedback**

RoboImagine learns autoregressive robotic world models by comparing candidate futures with semantic and visual feedback.

## Models and inference

Released weights: [easylearning/RoboImagine](https://huggingface.co/easylearning/RoboImagine).

```bash
pip install -e '.[notebook]'
roboimagine --dataset rt1 --variant main --example-index 0 --output predictions/rt1
```

[Inference guide](docs/INFERENCE.md) · [Run All inference notebook](notebooks/Inference.ipynb) · [Data and provenance](docs/DATA_AND_PROVENANCE.md)

The release includes RT-1, BridgeV2 and RoboCasa main/SFT checkpoints, codecs, text adapters and action ranges, plus two fixed demonstration examples per dataset. Model inference, fixed-anchor rolling memory, ordinary beam and GDBS implementations are included. The notebook runs prediction from released weights. Complete training orchestration and reward-service hosting are not included.

## Experimental reproducibility release

The [`experiments/`](experiments/) directory contains the code, saved results, and necessary example frames for **Figures 6–14 and Tables 1–5**. It covers continued-SFT comparisons, reward/search ablations, GDBS case analysis, and memory quality/efficiency. Tables 1–2 export RoboImagine rows only; external-baseline implementations and architectural drawing code are not included.

```bash
cd experiments
pip install -r requirements.txt
jupyter lab Reproduce_All.ipynb
# or
python run_all.py
```

CairoSVG requires system Cairo; table PDFs additionally require TeX/latexmk. See the [experiment guide](experiments/README.md) for figure-specific notebooks, protocols, dependencies, and limitations.

The `experiments/` notebooks redraw plots/tables **from archived experimental outputs**. The separate inference notebook runs predictions from released checkpoints. Neither path makes paid API calls or submits cluster jobs.

## Resources

- [Experiment code and guide](experiments/README.md)
- [Run All notebook](experiments/Reproduce_All.ipynb)
- [Data and weights hub](https://huggingface.co/easylearning/RoboImagine).

Third-party font/icon notices are retained with assets. No blanket license is granted over third-party datasets or brand marks; see [NOTICE](NOTICE.md).
