# Experiment reproduction / 实验复现

This directory contains experimental Figures **6–14** only. No model-architecture or method-schematic drawing code is included. It reproduces saved observations and statistics, not new model predictions.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab Reproduce_All.ipynb
# or all experimental figures:
python run_all.py
# selected figures:
python run_all.py --figures 8 11 14
# table previews (TeX + latexmk required):
python tables/build_tables.py
```

Use Python 3.10+; `requirements-tested.txt` records the locally tested library versions. CairoSVG requires system Cairo (`brew install cairo` on macOS or `libcairo2` on Linux); `CAIRO_LIBRARY` can point to the installed library if discovery fails. Output goes to ignored `reproduced/`. Run All does not overwrite the inputs, train, call an evaluator API, or connect to Ray. Each figure also has its own `Reproduce.ipynb`.

| Figure | 内容 | Script |
|---|---|---|
| 6 | GT 与模型预测的四时刻对比 | [build_figure.py](figures/fig06/build_figure.py) |
| 7 | 多模型评测与输出摘录 | [build_figure.py](figures/fig07/build_figure.py) |
| 8 | SFT 续训与搜索变体训练曲线 | [build_figure.py](figures/fig08/build_figure.py) |
| 9 | 训练 GPU 小时与质量 | [build_figure.py](figures/fig09/build_figure.py) |
| 10 | 三数据集 SFT 对照 | [render_figure.py](figures/fig10/render_figure.py) |
| 11 | 完整128条预测距离与反馈 | [build_figure.py](figures/fig11/build_figure.py) |
| 12 | 真实 GDBS 搜索案例 | [build_figure.py](figures/fig12/build_figure.py) |
| 13 | 64／128帧记忆对照 | [build_figure.py](figures/fig13/build_figure.py) |
| 14 | 窗口质量与推理效率 | [build_figure.py](figures/fig14/build_figure.py) |

## Protocol notes

- Figures 6/10: real saved frames, fixed time steps and within-case ROIs. Fig6 Bridge uses the validation-selected checkpoint. Fig10 RT-1 uses visual-only feedback; BridgeV2/RoboCasa use combined feedback.
- Fig7: 97 Qwen, 10 Doubao, and 50 GPT-6 Astra paired trajectories. GPT-6 Astra denotes a configured-model session review, not a version-pinned API evaluation. Individual excerpts are not aggregate means.
- Figs8/11 and search tables: full 128 RT-1 trajectories. Training search differs; evaluation decoding is shared. Fig11 compares final predictions between models, not candidates within one search.
- Fig12 is a selected training example, not an aggregate result.
- Fig13 retains the historical **11** held-out trajectories at64/128 frames.
- Fig14/Table5 use the same **30** long-sequence-eligible test trajectories at16/32/48/64/96 frames. Three timing repeats are averaged within each trajectory, not treated as90 independent trajectories. These means differ from the main128 cohort (W6/H32 PSNR22.20 vs22.57dB).
- Table1 values are author-provided aggregate Judge scores; this release does not reconstruct standard deviations from raw Judge responses. Table2 Bridge is the32-trajectory validation-selected result, not a matched128-trajectory test comparison.

## Contents and limitations

`tables/data/our_model_main_tables.csv` exports only RoboImagine main-table values. Table3/4/5 source summaries are retained. `input_checksums.json` checks the packaged inputs. Historical absolute paths and private server endpoints are removed from metadata. Hashes of source checkpoints and dataset IDs remain for provenance.

The public package does not include full model checkpoints, training data loaders, or an end-to-end inference pipeline. Public download adapters remain to be verified; do not interpret the notebook as rerunning the experiments from weights. Version-dependent font metrics may cause minor plot spacing changes; numerical data and original frames are unchanged.
