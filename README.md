<div align="center">

<img src="docs/assets/mark.svg" width="48" alt="RoboImagine">

# RoboImagine

### Learning Robotic World Models with Semantic and Visual Feedback

**Learning from comparisons between imagined futures.**

Hao Wu<sup>1</sup> · Yuqi Li<sup>2</sup> · Yizhou Zhao<sup>1</sup> · Kun Wang<sup>3</sup> · Xiao Luo<sup>4</sup> · Yu Cheng<sup>3</sup><br>
Yingli Tian<sup>2</sup> · Laszlo A. Jeni<sup>1</sup> · Ming-Hsuan Yang<sup>5</sup>

<sup>1</sup> Carnegie Mellon University · <sup>2</sup> The City College of New York<br>
<sup>3</sup> NTU · <sup>4</sup> University of Wisconsin–Madison · <sup>5</sup> UC Merced

[**Models & examples**](https://huggingface.co/easylearning/RoboImagine) · [**Inference notebook**](notebooks/Inference.ipynb) · [**Experiment code**](experiments/executed_sources/README.md) · [**简体中文**](README.zh-CN.md)

[Overview](#overview) · [Method](#method-overview) · [Results](#selected-results) · [Quick start](#quick-start) · [Reproducing the paper](#reproducing-the-paper)

</div>

## Overview

Accurate robotic prediction requires both **visual fidelity** and **task-relevant dynamics**. RoboImagine is a post-training framework for GPT-style autoregressive robotic world models that learns from comparisons among candidate futures conditioned on shared observations, actions, and instructions.

The framework connects three components:

- **Grouped diverse beam search (GDBS)** constructs distinct candidate trajectories for comparison.
- **Complementary semantic and visual feedback** evaluates task progress alongside pixel- and perceptual-level consistency.
- **History-consistent policy optimization** updates candidate probabilities using the same branch-specific context used during generation.

The release includes models for **RT-1, BridgeV2, and RoboCasa**, their SFT initializations, and the original programs underlying the paper’s training and evaluation experiments.

## Method overview

![RoboImagine: supervised initialization, diverse candidate search, complementary feedback, and policy optimization](docs/assets/method-overview.png)

RoboImagine first learns multimodal next-token prediction, then improves the model by comparing candidate futures:

1. **Supervised initialization.** A frozen visual codec quantizes scene and dynamics representations. A causal Transformer combines visual–action tokens with instruction features from a frozen T5 encoder; SFT trains the backbone and text cross-attention to predict future dynamics tokens.
2. **Grouped diverse beam search.** Four groups of two beams construct candidates under shared conditions. A soft token-overlap penalty encourages cross-group differences, with one complete trajectory retained per group.
3. **Complementary feedback.** The frozen 4B mJev evaluator (Jev) compares candidate videos in both orders using task context and eight sampled frames. Visual feedback is the negative sum of MAE and VGG-LPIPS. Candidate-wise standardized semantic and visual scores are fused with weights 0.25 and 0.75.
4. **History-consistent policy updates.** Group-relative advantages reweight token likelihoods under each candidate’s actual generation prefixes, with regularization toward the SFT policy. Fixed initial context and the observed dynamics anchor accompany six recent predicted state–action pairs. The window bounds input history, not output length.

**Inference.** The learned model uses frame-wise ordinary beam decoding of width four; neither the evaluator nor ground-truth futures is required. [Framework PDF](docs/assets/method-overview.pdf) · [Actual training and feedback code](experiments/executed_sources/rt1_hybrid/README.md)

## Selected results

### Prediction quality across three datasets

RoboImagine’s reported 32-frame prediction results (Table 2):

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:--|--:|--:|--:|
| RT-1 | **22.57** | **0.8035** | **0.1541** |
| BridgeV2 | **21.53** | **0.7880** | **0.1527** |
| RoboCasa | **18.39** | **0.7727** | **0.1768** |

*RT-1 and RoboCasa use 128 reporting trajectories each; BridgeV2 uses the labeled 32-trajectory validation cohort and its validation-selected checkpoint.* [Table 2 and sources](experiments/tables/README.md)

### Gains from post-training

**Controlled post-training comparison on RT-1.** All rows below share the same SFT initialization and evaluation protocol: 128 trajectories, 32 predicted frames, and the same decoder.

| Model | Additional updates | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:--|--:|--:|--:|--:|
| SFT initialization | 0 | 21.37 | 0.7860 | 0.1734 |
| Continued SFT | 500 | 21.31 | 0.7844 | 0.1750 |
| **RoboImagine** | **500** | **22.57** | **0.8035** | **0.1541** |

At the same update count, RoboImagine improves PSNR by **1.26 dB** and reduces LPIPS by **11.93%** relative to continued SFT. Gains are computed from unrounded measurements. Equal update counts do not imply equal computation; training GPU-hours are reported separately.

[Checkpoint results and protocol](experiments/figures/fig08/README.md) · [Training cost](experiments/figures/fig09/README.md) · [Full experiment index](docs/PAPER_INDEX.md)

<table>
<tr><th>Ground truth</th><th>SFT initialization</th><th>Continued SFT</th><th>RoboImagine</th></tr>
<tr>
<td><img src="docs/assets/drawer-gt.png" alt="Ground-truth drawer scene"></td>
<td><img src="docs/assets/drawer-initial_sft.png" alt="SFT prediction"></td>
<td><img src="docs/assets/drawer-sft_500.png" alt="Continued-SFT prediction"></td>
<td><img src="docs/assets/drawer-hybrid_500.png" alt="RoboImagine prediction"></td>
</tr>
</table>

*One drawer-closing example from Figure 8: all images show prediction frame +32. Continued SFT and RoboImagine each use 500 additional updates. This qualitative example is separate from the aggregate results.*

### Semantic evaluation and feedback ablation

An independent Qwen3-VL-8B-Thinking evaluator scores task-related quality on **97 shared valid RT-1 trajectories** (0–10; Table 3).

| Configuration | Qwen total ↑ |
|:--|--:|
| SFT initialization | 7.04 |
| Semantic feedback only | 7.69 |
| Visual feedback only | 7.97 |
| **RoboImagine · combined feedback** | **8.30** |

Combined feedback achieves the highest semantic-evaluation score among these configurations, while visual-only feedback yields the strongest reconstruction metrics in this ablation. The two criteria capture complementary aspects of prediction quality. [Ablation results and scoring protocol](experiments/tables/README.md)

## Quick start

Use **Python 3.10+** in a Linux CUDA environment. Install a compatible PyTorch/torchvision pair first, then:

```bash
git clone https://github.com/Alexander-wu/RoboImagine.git
cd RoboImagine
python -m pip install -e '.[notebook,evaluation]'

# Download the released model, generate a video, and compute visual metrics.
roboimagine --dataset rt1 --variant main --example-index 0 \
  --metrics --output predictions/rt1

# Compare with the matching SFT initialization.
roboimagine --dataset rt1 --variant sft --example-index 0 \
  --metrics --output predictions/rt1_sft
```

The first run downloads pinned model assets and the frozen T5 encoder; optional LPIPS evaluation also requires VGG weights. Outputs include `prediction.mp4`, float predictions and tokens, `memory_trace.json`, and `run.json` with the configuration and computed metrics.

**[Inference guide (English)](docs/INFERENCE.md) · [推理教程（中文）](docs/INFERENCE.zh-CN.md) · [Inference notebook](notebooks/Inference.ipynb)**

### Model zoo

| Dataset | Main checkpoint | Matching initialization | Examples |
|---|---|---|---|
| RT-1 | Hybrid, 500 updates | Text-enabled SFT | 2 examples, 32 predicted frames |
| BridgeV2 | Hybrid, validation-selected checkpoint | Matching text-adapted SFT | 2 examples, 32 predicted frames |
| RoboCasa | Hybrid, 500 updates | Text-enabled SFT | 2 examples, 32 predicted frames |

All six predictors, codecs, adapters, action ranges, and demonstration inputs are available on **[Hugging Face](https://huggingface.co/easylearning/RoboImagine)**. Use `--dataset bridge` or `--dataset robocasa` to switch datasets and `--variant sft` for the corresponding initialization. Checkpoint selection and evaluation protocols are documented in the [model guide](docs/INFERENCE.md#scope).

```bash
# Inspect how a shorter history changes inference.
roboimagine --dataset rt1 --window 3 --output predictions/rt1_w3

# Use your own observations, actions, and instruction.
roboimagine --dataset rt1 --input case.npz --horizon 32 --output predictions/custom
```

See the [input specification](docs/INFERENCE.md#your-own-inputs) for shapes, action layouts, and longer rollouts.

## Reproducing the paper

The repository provides three connected entry points: **model inference** from released checkpoints, **original experiment programs** used on the server, and **figure/table rendering** from archived outputs. Start with the workflow relevant to your goal.

| Workflow | Entry point | What it provides |
|:--|:--|:--|
| Model inference | [Guide](docs/INFERENCE.md) · [Notebook](notebooks/Inference.ipynb) | Checkpoint loading, autoregressive generation, memory traces, and metric computation |
| Training and evaluation | [Executed sources](experiments/executed_sources/README.md) | Training loops, search, reward computation, prediction, scoring, and aggregation |
| Figure and table inspection | [Paper-to-code index](docs/PAPER_INDEX.md) | Per-figure inputs, scripts, notebooks, and source records |
| Figure rendering | [Experiment guide](experiments/README.md) · [Run All notebook](experiments/Reproduce_All.ipynb) | Visualizations reconstructed from archived experiment outputs |

Original experiment guides record the configurations, data preparation, dependencies, and paths to adapt for a new environment. The inference notebook runs the model; figure notebooks render existing experiment records.

### Experiments and ablations

| Research question | Evidence | Implementation |
|:--|:--|:--|
| **RQ1 · Prediction quality and task semantics** | Visual metrics, multidimensional evaluation, and qualitative comparisons across RT-1, BridgeV2, and RoboCasa | [Tables 1–2](experiments/tables/README.md) · [Figures 6–7](docs/PAPER_INDEX.md#figure-by-figure-navigation) |
| **RQ2 · Does post-training improve over continued SFT?** | Shared-initialization controls, checkpoint curves, and training GPU-hours | [Continued SFT](experiments/executed_sources/continued_sft/README.md) · [Learning curves](experiments/figures/fig08/README.md) · [Compute accounting](experiments/figures/fig09/README.md) |
| **RQ3 · What do feedback and search contribute?** | Semantic-only, visual-only, and combined feedback; sampling, ordinary beam, GDBS, and likelihood-guided MCTS | [Search ablations](experiments/executed_sources/search_ablation/README.md) · [MCTS](experiments/executed_sources/mcts/README.md) · [Tables 3–4](experiments/tables/README.md) |
| **RQ4 · How does bounded history affect quality and efficiency?** | Matched-trajectory window sensitivity, long-horizon prediction, latency, and peak GPU memory | [Window study](experiments/executed_sources/window30/README.md) · [Figures 13–14](docs/PAPER_INDEX.md#figure-by-figure-navigation) |

<details>
<summary><strong>Evaluation protocols and release scope</strong></summary>

- The controlled RT-1 results above use 128 trajectories and 32 predicted frames. Qwen feedback/search ablations use 97 paired valid trajectories.
- Table 2’s BridgeV2 result is reported on its labeled 32-trajectory validation cohort.
- Table 5 and Figure 14 use the same 30 long-eligible test trajectories across five horizons. Figure 13 retains the separate historical 11-trajectory study. These cohorts are not interchangeable.
- Main and matching SFT checkpoints, codecs, adapters, and demonstration inputs are public. Full prepared evaluation cohorts, evaluator assets, and all intermediate/ablation checkpoints are not bundled. The [executed-source guide](experiments/executed_sources/README.md#portability-provenance-and-remaining-assets) lists the requirements for full runs.

</details>

**Documentation:** [Inference EN](docs/INFERENCE.md) / [中文](docs/INFERENCE.zh-CN.md) · [Experiments EN](experiments/README.md) / [中文](experiments/README.zh-CN.md) · [Executed sources EN](experiments/executed_sources/README.md) / [中文](experiments/executed_sources/README.zh-CN.md)

## Repository structure

```text
src/roboimagine/                Public inference, model, search, memory, and metrics
model_configs/                 Dataset-specific release configurations
notebooks/Inference.ipynb       Model inference notebook
experiments/executed_sources/  Original training and evaluation programs
experiments/figures/           Experimental Figures 6–14: inputs, code, guides
experiments/tables/            Tables 1–5: sources and rendering
provenance/                    Release manifests and verification records
docs/                          Inference guides, paper index, and figure assets
```

## Release verification and attribution

The release includes [model validation](provenance/model_validation.json), [fresh-cache download checks](provenance/public_download_validation.json), and a [release manifest](provenance/release_manifest.json). Recorded checks cover main-model prediction parity and matching SFT smoke tests in the documented CUDA environment.

See [data provenance](docs/DATA_AND_PROVENANCE.md) and [source and asset notices](NOTICE.md) for dataset origins and third-party terms. Implementation and reproduction questions are welcome through [GitHub Issues](https://github.com/Alexander-wu/RoboImagine/issues).
