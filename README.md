# RoboImagine

**Learning Robotic World Models with Semantic and Visual Feedback**

RoboImagine learns autoregressive robotic world models by comparing candidate futures with semantic and visual feedback.

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

This release reproduces plots and tables **from archived experimental outputs**. It does not yet provide end-to-end training or checkpoint inference. There are no automatic paid API calls or cluster submissions.

## Resources

- [Experiment code and guide](experiments/README.md)
- [Run All notebook](experiments/Reproduce_All.ipynb)
- [Data and weights hub](https://huggingface.co/easylearning/RoboImagine) — checkpoint release remains in preparation.

Third-party font/icon notices are retained with assets. No blanket license is granted over third-party datasets or brand marks; see [NOTICE](NOTICE.md).
