# Paper-to-code navigation

[Project](../README.md) · [中文](../experiments/README.zh-CN.md) · [Executed experiment sources](../experiments/executed_sources/README.md)

Every entry below connects an experimental figure or table to its data, plotting code, and guide. For training, inference, rewards and raw scoring programs, start with the [executed-source map](../experiments/executed_sources/README.md#paper-to-source-map).

### Figure-by-figure navigation

| Paper figure | Content | Guide | Notebook | Code |
|---|---|---|---|---|
| 6 | Four-timestamp prediction comparison | [EN](../experiments/figures/fig06/README.md) / [中文](../experiments/figures/fig06/README.zh-CN.md) | [Run All](../experiments/figures/fig06/Reproduce.ipynb) | [Script](../experiments/figures/fig06/build_figure.py) |
| 7 | Multi-model evaluation and excerpts | [EN](../experiments/figures/fig07/README.md) / [中文](../experiments/figures/fig07/README.zh-CN.md) | [Run All](../experiments/figures/fig07/Reproduce.ipynb) | [Script](../experiments/figures/fig07/build_figure.py) |
| 8 | Continued-SFT and search-variant learning curves | [EN](../experiments/figures/fig08/README.md) / [中文](../experiments/figures/fig08/README.zh-CN.md) | [Run All](../experiments/figures/fig08/Reproduce.ipynb) | [Script](../experiments/figures/fig08/build_figure.py) |
| 9 | Training cost and prediction quality | [EN](../experiments/figures/fig09/README.md) / [中文](../experiments/figures/fig09/README.zh-CN.md) | [Run All](../experiments/figures/fig09/Reproduce.ipynb) | [Script](../experiments/figures/fig09/build_figure.py) |
| 10 | Three-dataset SFT comparison | [EN](../experiments/figures/fig10/README.md) / [中文](../experiments/figures/fig10/README.zh-CN.md) | [Run All](../experiments/figures/fig10/Reproduce.ipynb) | [Script](../experiments/figures/fig10/render_figure.py) |
| 11 | Prediction distances and feedback | [EN](../experiments/figures/fig11/README.md) / [中文](../experiments/figures/fig11/README.zh-CN.md) | [Run All](../experiments/figures/fig11/Reproduce.ipynb) | [Script](../experiments/figures/fig11/build_figure.py) |
| 12 | GDBS search case study | [EN](../experiments/figures/fig12/README.md) / [中文](../experiments/figures/fig12/README.zh-CN.md) | [Run All](../experiments/figures/fig12/Reproduce.ipynb) | [Script](../experiments/figures/fig12/build_figure.py) |
| 13 | Long-horizon memory comparison | [EN](../experiments/figures/fig13/README.md) / [中文](../experiments/figures/fig13/README.zh-CN.md) | [Run All](../experiments/figures/fig13/Reproduce.ipynb) | [Script](../experiments/figures/fig13/build_figure.py) |
| 14 | Window sensitivity and inference efficiency | [EN](../experiments/figures/fig14/README.md) / [中文](../experiments/figures/fig14/README.zh-CN.md) | [Run All](../experiments/figures/fig14/Reproduce.ipynb) | [Script](../experiments/figures/fig14/build_figure.py) |

### Table-by-table navigation

| Paper table | Content | Source data | Reproduction |
|---|---|---|---|
| 1 | Six-dimensional RoboAlign-Judge scores | [Our reported rows](../experiments/tables/data/our_model_main_tables.csv) | [EN](../experiments/tables/README.md) / [中文](../experiments/tables/README.zh-CN.md) |
| 2 | RT-1, BridgeV2 and RoboCasa visual quality | [Our reported rows](../experiments/tables/data/our_model_main_tables.csv) | [Table 2 source](../experiments/tables/table02.tex) |
| 3 | Feedback composition and candidate-search ablations | [Visual](../experiments/tables/data/table03_visual_results.json) / [Qwen](../experiments/tables/data/table03_qwen97_results.json) | [Table 3 source](../experiments/tables/table03.tex) |
| 4 | Sampling, MCTS, ordinary beam, GDBS and training cost | [Full results](../experiments/tables/data/table04_search_results.json) | [Table 4 source](../experiments/tables/table04.tex) |
| 5 | History-window quality and inference efficiency | [Matched-30 results](../experiments/tables/data/table05_matched30_results.json) | [Table 5 source](../experiments/tables/table05.tex) |

Tables 1–2 export our model rows only, without external-baseline rankings. The other tables retain the relevant SFT controls and within-model variants. The [bilingual table guide](../experiments/tables/README.md) explains each cohort and source.

