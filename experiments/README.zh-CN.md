# 论文实验复现教程

[English](README.md) | [简体中文](README.zh-CN.md) | [项目主页](../README.md)

本目录对应论文**图 6–14、表 1–5**，包括新增的反馈、搜索、续训和记忆实验。模型结构示意图的绘图代码不在本次实验发布范围内。

## 目前支持哪些复现？

| 任务 | 入口 | 范围 |
|---|---|---|
| 重绘论文结果图 | 各图 notebook / `run_all.py` | 读取保存的实验输出，无需 GPU 或评价 API |
| 重新生成表格 | `tables/build_tables.py` | 五页独立预览，表 1–2 仅导出我们模型的行 |
| 从公开权重生成预测 | [推理教程](../docs/INFERENCE.zh-CN.md)、[推理 notebook](../notebooks/Inference.ipynb) | 三数据集主模型及 SFT，共六套预测权重、六个演示输入 |
| 尝试不同历史窗口 | 推理 CLI 的 `--window` / `--no-anchor` | 对给定输入重新推理，不重新训练 |
| 从头训练全部消融、重跑所有评价器 | 尚未提供完整端到端流程 | 不包含训练调度、奖励服务托管、完整预处理评估集及消融/中间检查点 |

模型实现包含普通 beam 和 GDBS；MCTS 目前发布的是实验结果与协议，不包含可执行的 MCTS 训练入口。推理 CLI 的评估 beam width **不等于**训练阶段的搜索消融选项。

## 安装与一键运行

在仓库根目录创建 Python 3.10+ 环境：

```bash
python -m venv .venv-plots
source .venv-plots/bin/activate
python -m pip install -r experiments/requirements.txt
cd experiments
jupyter lab Reproduce_All.ipynb
```

选择上述环境的内核，执行 **Run All**。[总 notebook](Reproduce_All.ipynb) 重绘图 6–14，并尝试编译表 1–5。也可以在 `experiments/` 下运行：

```bash
python run_all.py
python tables/build_tables.py
# 只重绘部分图：
python run_all.py --figures 8 11 14
```

CairoSVG 需要系统 Cairo：macOS 可用 `brew install cairo`，Linux 安装 `libcairo2`；无法自动发现时可设置 `CAIRO_LIBRARY`。表格 PDF 另需 TeX、`latexmk` 及 [table_style.tex](tables/table_style.tex) 所列包。缺少 TeX 时表格构建会记录跳过，不能视为编译成功，但不影响图的重绘。[requirements-tested.txt](requirements-tested.txt) 保存此前本地绘图环境版本，不保证各平台渲染逐字节一致。

输出位于 `experiments/reproduced/`：各图 PDF 和日志在 `figures/figNN/`，表格在 `tables/tables.pdf`，另有执行回执。图形入口运行前后检查 [input_checksums.json](input_checksums.json)，不覆盖输入数据。字体或渲染库版本可能影响细微间距。绘图 notebook 不提交 Ray 任务、不训练模型、不调用付费 API。

## 论文图索引

| 论文图 | 内容 | 教程 | Notebook | 代码 |
|---|---|---|---|---|
| 6 | GT 与模型预测的四时刻对比 | [EN](figures/fig06/README.md) / [中文](figures/fig06/README.zh-CN.md) | [Run All](figures/fig06/Reproduce.ipynb) | [Script](figures/fig06/build_figure.py) |
| 7 | 多模型评测与输出摘录 | [EN](figures/fig07/README.md) / [中文](figures/fig07/README.zh-CN.md) | [Run All](figures/fig07/Reproduce.ipynb) | [Script](figures/fig07/build_figure.py) |
| 8 | SFT 续训与搜索变体训练曲线 | [EN](figures/fig08/README.md) / [中文](figures/fig08/README.zh-CN.md) | [Run All](figures/fig08/Reproduce.ipynb) | [Script](figures/fig08/build_figure.py) |
| 9 | 训练 GPU 小时与质量 | [EN](figures/fig09/README.md) / [中文](figures/fig09/README.zh-CN.md) | [Run All](figures/fig09/Reproduce.ipynb) | [Script](figures/fig09/build_figure.py) |
| 10 | 三数据集 SFT 对照 | [EN](figures/fig10/README.md) / [中文](figures/fig10/README.zh-CN.md) | [Run All](figures/fig10/Reproduce.ipynb) | [Script](figures/fig10/render_figure.py) |
| 11 | 完整128条预测距离与反馈 | [EN](figures/fig11/README.md) / [中文](figures/fig11/README.zh-CN.md) | [Run All](figures/fig11/Reproduce.ipynb) | [Script](figures/fig11/build_figure.py) |
| 12 | 真实 GDBS 搜索案例 | [EN](figures/fig12/README.md) / [中文](figures/fig12/README.zh-CN.md) | [Run All](figures/fig12/Reproduce.ipynb) | [Script](figures/fig12/build_figure.py) |
| 13 | 64／128帧记忆对照 | [EN](figures/fig13/README.md) / [中文](figures/fig13/README.zh-CN.md) | [Run All](figures/fig13/Reproduce.ipynb) | [Script](figures/fig13/build_figure.py) |
| 14 | 窗口质量与推理效率 | [EN](figures/fig14/README.md) / [中文](figures/fig14/README.zh-CN.md) | [Run All](figures/fig14/Reproduce.ipynb) | [Script](figures/fig14/build_figure.py) |

## 论文表索引

| 表 | 对应实验 | 数据与教程 |
|---|---|---|
| 1 | RoboAlign-Judge 六维评分 | [主表数据](tables/data/our_model_main_tables.csv)、[表格教程](tables/README.zh-CN.md) |
| 2 | 三数据集视觉预测指标 | [主表数据](tables/data/our_model_main_tables.csv)、[表 2 源码](tables/table02.tex) |
| 3 | 反馈组成与训练候选搜索 | [视觉指标](tables/data/table03_visual_results.json)、[Qwen 评分](tables/data/table03_qwen97_results.json) |
| 4 | 完整搜索对比及训练成本 | [数据和协议](tables/data/table04_search_results.json)、[表 4 源码](tables/table04.tex) |
| 5 | 窗口敏感性及效率 | [同 30 条结果](tables/data/table05_matched30_results.json)、[图 14](figures/fig14/README.zh-CN.md) |

表 1 的均值和偏差来自作者提供的汇总数据，本发布包不从原始 Judge 响应重算偏差。表 2 的 RT-1/RoboCasa 使用各自 128 条报告轨迹；Bridge 使用 32 条验证轨迹及验证集选择的检查点。公开演示案例不替代这些完整评估队列。

## 消融实验：改变什么，保持什么？

### 反馈组成——表 3(A)

对比 SFT 初始化、仅语义反馈（Jev）、仅视觉反馈、混合反馈（Hybrid）。后训练版本共享初始化、500 步更新、RT-1 报告轨迹和评估解码，训练候选由 GDBS 生成。视觉指标对 128 条轨迹的 32 个预测帧统计。这是现有单次训练记录下的反馈对比，不是多随机种子实验。

**A 块的 Hybrid 与 B 块的 GDBS 是同一配置**，其视觉指标和当前配对 Qwen 得分应一致。当前 Qwen 文件已覆盖六个配置的**同 97 条有效配对轨迹**，替代旧的 9 条搜索试验。随机采样配置中三条格式异常响应进行了有上限的格式修复重试，源 JSON 同时保留原协议与修复记录。独立评分模型 Qwen3-VL-8B-Thinking 与训练用的冻结 Jev 是不同评价器。

### 候选搜索——表 3(B)、表 4、图 8/11/12

随机采样、普通 beam 和 GDBS 改变训练候选的构造方式，保持混合反馈。GDBS 使用 4 组、每组 2 个 beam，每组保留一条完整轨迹；普通 beam 使用全局候选池。表 4 另加入已实现的似然引导 MCTS，使用最终 500 步模型在完整 128 条轨迹上评估，已替代早期 8 条、100 步的初步结果。

MCTS 采用帧前缀 UCT、16 次 rollout、分支数 2、探索系数 1，按原始对数概率返回 4 条轨迹；树搜索阶段不使用真实未来或 VLM 奖励，选出候选后才计算反馈用于策略训练。这一特定 MCTS 配置不代表所有 MCTS 方法，其计算预算也不同于 beam。详见 [table04_search_results.json](tables/data/table04_search_results.json)。

图 11 展示**不同模型最终预测之间**的 token 距离及训练评价器反馈；图 12 展示**一次搜索内部**实际保留分支的祖先关系。更大 token 距离或某个案例的优势本身不等于总体质量提升。

### 额外训练与后训练——图 8/9

SFT 续训与 RL 版本从相同 SFT 初始化开始。图 8 使用实际评测的 100/250/500 步检查点和相同解码协议。图 9 的成本为分配 GPU 数乘训练循环墙钟时间，包含循环内搜索及反馈开销。表 4 再除以更新次数乘全局 batch，得到每次轨迹**呈现**的训练 GPU 秒，而不是每条独立轨迹的成本。此处训练成本与表 5/图 14 的预热推理延迟不同。

### 有界历史——表 5、图 13/14

此实验不重新训练，固定模型权重，只改变最近历史窗口。窗口 W 不含固定动态锚点；初始上下文始终保留。原始网格包含仅当前帧，以及保留锚点的 W=1/3/6/12。

表 5/图 14 在 H=16/32/48/64/96 下使用同 30 条 test 轨迹：从至少有 98 个真实帧且带指令的 38 条合格长序列中，以 seed=20261026 选取。完整结果包含 750 条质量记录、2,250 条生成计时记录。每条轨迹的三次计时先平均，再对 30 条轨迹平均，不能当成 90 条独立轨迹；质量先对帧平均，再对轨迹平均。汇总文件包含单检查点下、10,000 次配对轨迹 bootstrap 的 95% 区间。

预热后的完整 rollout 时间不含模型加载、文本缓存构建、评分和文件 I/O。显存为包含常驻模型的 PyTorch allocated memory，不是 reserved VRAM。使用 beam width=4、FP32、关闭 TF32，实验记录核验 GPU 独占情况。

**本组 W6/H32 的 22.20 dB 来自 30 条长序列合格轨迹，主表 22.57 dB 来自原来的 128 条，两者均值不能直接比较。** 图 13 仍为历史同 11 条 held-out 轨迹的 64/128 帧实验，既不是这次同 30 条实验，也不能独立证明锚点的贡献。

## 从公开权重生成新结果

请使用[推理教程](../docs/INFERENCE.zh-CN.md)。公开包提供三个数据集的主模型/SFT，以及每个数据集两个 H32 案例；完整评估输入、中间搜索检查点、仅语义/仅视觉奖励及 MCTS/普通 beam 后训练权重尚未包含。更长预测需要足够长度的真实动作，六个演示输入不能复现上述完整队列均值。

## 完整性与素材归属

[figure_index.json](figure_index.json) 记录图号、脚本及输出名，[input_checksums.json](input_checksums.json) 固定绘图输入哈希，[发布溯源记录](../provenance/) 保存权重验证与下载检查。原始权重哈希和实验标识在可用时保留，运行无需私有服务器地址或凭证。素材遵循各自字体/图标说明与 [NOTICE](../NOTICE.md)，本项目不统一授权第三方数据集或基线实现。
