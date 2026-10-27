# RoboImagine

**Learning Robotic World Models with Semantic and Visual Feedback**

RoboImagine learns autoregressive robotic world models by comparing candidate futures with semantic and visual feedback.

[Models & examples on Hugging Face](https://huggingface.co/easylearning/RoboImagine) · [Experiment guide (English)](experiments/README.md) · [实验复现教程（中文）](experiments/README.zh-CN.md) · [Inference guide](docs/INFERENCE.md) · [推理教程（中文）](docs/INFERENCE.zh-CN.md)

## Reproduce predictions and inspect the actual experiment code

**Start with [the executed experiment sources](experiments/executed_sources/README.md) / [服务器原始实验代码与中文教程](experiments/executed_sources/README.zh-CN.md).** These are the training, candidate-search, video prediction, visual metrics, judge and aggregation programs retrieved from the already completed server runs. They are not calculators populated with paper means.

| Goal | Entry point | What it does |
|---|---|---|
| Generate and score a prediction from public weights | [Inference guide](docs/INFERENCE.md) · [notebook](notebooks/Inference.ipynb) | Loads the checkpoint; generates pixels; optional `--metrics` compares predictions with GT |
| Inspect or adapt the actual paper experiment | [Paper-to-source map](experiments/executed_sources/README.md#paper-to-source-map) · [中文](experiments/executed_sources/README.zh-CN.md) | Original training/inference/scoring/aggregation code and protocols |
| Trace feedback, MCTS/beam, continued-SFT and window ablations | [Computation entry points](experiments/executed_sources/README.md#start-with-the-computation) | Actual algorithm implementations, prompts, averaging and timing logic |
| Redraw the archived figures / tables | [Plot notebook](experiments/Reproduce_All.ipynb) | Visualization only; does not rerun or validate an experiment |

Full-cohort retraining still needs the corresponding prepared datasets, evaluator assets and original run inputs listed in the [source guide](experiments/executed_sources/README.md#portability-provenance-and-remaining-assets). The released main/SFT demos already have a separate public inference interface. Plotting saved results must not be confused with reproducing model results.

## 1. Redraw archived experimental figures and tables

Clone the repository and create a Python 3.10+ environment:

```bash
git clone https://github.com/Alexander-wu/RoboImagine.git
cd RoboImagine
python -m venv .venv-plots
source .venv-plots/bin/activate
python -m pip install -r experiments/requirements.txt
cd experiments
jupyter lab Reproduce_All.ipynb
```

Select the installed environment's kernel and choose **Run All**. For command-line **redrawing**, from `experiments/`:

```bash
python run_all.py                         # Figures 6–14
python tables/build_tables.py             # Tables 1–5
python run_all.py --figures 8 11 14        # Optional subset
```

Results appear in `experiments/reproduced/`, with per-figure logs and execution receipts. Inputs are hash-checked before and after replay. CairoSVG requires system Cairo; table previews additionally require TeX/latexmk. Missing TeX is recorded as a skip, not successful compilation. See [installation details](experiments/README.md#installation-and-run-all) or [中文安装说明](experiments/README.zh-CN.md#安装与一键运行).

### Figure-by-figure navigation

| Paper figure | Content | Guide | Notebook | Code |
|---|---|---|---|---|
| 6 | Four-timestamp prediction comparison | [EN](experiments/figures/fig06/README.md) / [中文](experiments/figures/fig06/README.zh-CN.md) | [Run All](experiments/figures/fig06/Reproduce.ipynb) | [Script](experiments/figures/fig06/build_figure.py) |
| 7 | Multi-model evaluation and excerpts | [EN](experiments/figures/fig07/README.md) / [中文](experiments/figures/fig07/README.zh-CN.md) | [Run All](experiments/figures/fig07/Reproduce.ipynb) | [Script](experiments/figures/fig07/build_figure.py) |
| 8 | Continued-SFT and search-variant learning curves | [EN](experiments/figures/fig08/README.md) / [中文](experiments/figures/fig08/README.zh-CN.md) | [Run All](experiments/figures/fig08/Reproduce.ipynb) | [Script](experiments/figures/fig08/build_figure.py) |
| 9 | Training cost and prediction quality | [EN](experiments/figures/fig09/README.md) / [中文](experiments/figures/fig09/README.zh-CN.md) | [Run All](experiments/figures/fig09/Reproduce.ipynb) | [Script](experiments/figures/fig09/build_figure.py) |
| 10 | Three-dataset SFT comparison | [EN](experiments/figures/fig10/README.md) / [中文](experiments/figures/fig10/README.zh-CN.md) | [Run All](experiments/figures/fig10/Reproduce.ipynb) | [Script](experiments/figures/fig10/render_figure.py) |
| 11 | Prediction distances and feedback | [EN](experiments/figures/fig11/README.md) / [中文](experiments/figures/fig11/README.zh-CN.md) | [Run All](experiments/figures/fig11/Reproduce.ipynb) | [Script](experiments/figures/fig11/build_figure.py) |
| 12 | GDBS search case study | [EN](experiments/figures/fig12/README.md) / [中文](experiments/figures/fig12/README.zh-CN.md) | [Run All](experiments/figures/fig12/Reproduce.ipynb) | [Script](experiments/figures/fig12/build_figure.py) |
| 13 | Long-horizon memory comparison | [EN](experiments/figures/fig13/README.md) / [中文](experiments/figures/fig13/README.zh-CN.md) | [Run All](experiments/figures/fig13/Reproduce.ipynb) | [Script](experiments/figures/fig13/build_figure.py) |
| 14 | Window sensitivity and inference efficiency | [EN](experiments/figures/fig14/README.md) / [中文](experiments/figures/fig14/README.zh-CN.md) | [Run All](experiments/figures/fig14/Reproduce.ipynb) | [Script](experiments/figures/fig14/build_figure.py) |

### Table-by-table navigation

| Paper table | Content | Source data | Reproduction |
|---|---|---|---|
| 1 | Six-dimensional RoboAlign-Judge scores | [Our reported rows](experiments/tables/data/our_model_main_tables.csv) | [EN](experiments/tables/README.md) / [中文](experiments/tables/README.zh-CN.md) |
| 2 | RT-1, BridgeV2 and RoboCasa visual quality | [Our reported rows](experiments/tables/data/our_model_main_tables.csv) | [Table 2 source](experiments/tables/table02.tex) |
| 3 | Feedback composition and candidate-search ablations | [Visual](experiments/tables/data/table03_visual_results.json) / [Qwen](experiments/tables/data/table03_qwen97_results.json) | [Table 3 source](experiments/tables/table03.tex) |
| 4 | Sampling, MCTS, ordinary beam, GDBS and training cost | [Full results](experiments/tables/data/table04_search_results.json) | [Table 4 source](experiments/tables/table04.tex) |
| 5 | History-window quality and inference efficiency | [Matched-30 results](experiments/tables/data/table05_matched30_results.json) | [Table 5 source](experiments/tables/table05.tex) |

Tables 1–2 export our model rows only, without external-baseline rankings. The other tables retain the relevant SFT controls and within-model variants. The [bilingual table guide](experiments/tables/README.md) explains each cohort and source.

## 2. Generate predictions from released models

Use a separate Python 3.10+ environment with a PyTorch/torchvision installation compatible with your CUDA system. From the repository root:

```bash
python -m pip install -e '.[notebook]'
roboimagine --dataset rt1 --variant main --example-index 0 --output predictions/rt1
roboimagine --dataset bridge --variant main --example-index 0 --output predictions/bridge
roboimagine --dataset robocasa --variant main --example-index 0 --output predictions/robocasa
# Matching SFT initialization:
roboimagine --dataset rt1 --variant sft --example-index 0 --output predictions/rt1_sft
# Interactive alternative:
jupyter lab notebooks/Inference.ipynb
```

The loader downloads pinned files from [easylearning/RoboImagine](https://huggingface.co/easylearning/RoboImagine), plus the pinned frozen T5 encoder. The release contains six main/SFT predictors, codecs, text adapters, action ranges and two fixed H32 demonstration inputs per dataset. Outputs include a prediction video, original float pixels/tokens, memory trace and run metadata. No evaluator API is called.

RT-1 and RoboCasa main models use 500 post-training updates. Bridge uses the validation-selected 250-update checkpoint and its matching text-enabled SFT initialization. For exact input shapes, optional SSIM/LPIPS, offline paths and limitations, see [English inference instructions](docs/INFERENCE.md) / [中文推理教程](docs/INFERENCE.zh-CN.md).

### Try bounded memory

```bash
roboimagine --dataset rt1 --variant main --window 3 --output predictions/rt1_w3
roboimagine --dataset rt1 --variant main --no-anchor --window 1 --output predictions/rt1_current
```

`--window` controls recent history, excluding the fixed dynamic anchor. Output horizon is a separate parameter; bundled examples support 32 predictions. These demo commands test inference behavior on individual examples, not the full Table 5 evaluation cohort.

## 3. Understand the newly added ablations

- **Feedback:** Table 3(A) compares semantic-only, visual-only and Hybrid against SFT. Hybrid and the GDBS row in Table 3(B) are the same configuration. Current Qwen scores share 97 paired valid trajectories across all six arms; the earlier nine-trajectory pilot is superseded.
- **Training candidate search:** Table 3(B)/4 compare sampling, ordinary beam, GDBS and the specified MCTS variant. Table 4 uses final 500-update checkpoints on all 128 reporting trajectories. Figure 11 compares final predictions between models; Figure 12 shows an actual search case.
- **Extra training and cost:** Figures 8/9 contrast continued SFT and post-training, separating equal update counts from actual training GPU-hours. Intermediate checkpoints are archived in the experimental results, but not all are published as downloadable weights.
- **History and efficiency:** Table 5/Figure 14 use the same 30 long-eligible test trajectories at five horizons; Figure 13 remains the separate historical 11-trajectory long-horizon experiment. Three timing repeats are not three independent trajectories.

Full settings, cohort definitions, averaging rules, costs and current release coverage are in the [English experiment guide](experiments/README.md) and [中文版](experiments/README.zh-CN.md). In particular, the matched-30 W6/H32 PSNR (22.20 dB) and original 128-trajectory main result (22.57 dB) average different cohorts. Table 2's Bridge result uses 32 validation trajectories; the demo inputs do not recreate that mean.

## Release scope and verification

Model inference and the original executed training, MCTS/beam search, reward-service, metric and aggregation sources are included. The archived worker sources retain their original protocol gates and require relocation/input preparation; they are not a portable all-experiment launcher. Full prepared cohorts, evaluator assets and every ablation/intermediate checkpoint are not yet bundled. Architecture/method illustration scripts are outside this release.

The six main-model demonstration outputs match archived token sequences and float pixels in the tested CUDA environment. Matching SFT smoke tests, notebook execution and a separate fresh-cache public-download check are recorded in [model validation](provenance/model_validation.json), [download validation](provenance/public_download_validation.json) and the [release manifest](provenance/release_manifest.json). These checks validate the included examples, not every paper trajectory.

[Data and provenance](docs/DATA_AND_PROVENANCE.md) · [Input checksums](experiments/input_checksums.json) · [Asset notices](NOTICE.md)

Third-party font/icon notices are retained. This release does not grant blanket rights over third-party datasets or brand marks.
