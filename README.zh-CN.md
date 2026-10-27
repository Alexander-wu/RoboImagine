<div align="center">

<img src="docs/assets/mark.svg" width="48" alt="RoboImagine">

# RoboImagine

### Learning Robotic World Models with Semantic and Visual Feedback

**比较候选未来，通过语义与视觉反馈学习机器人世界模型。**

[方法框架](#方法框架) · [结果概览](#结果概览) · [模型与样例](https://huggingface.co/easylearning/RoboImagine) · [快速开始](#快速开始) · [原始实验代码](experiments/executed_sources/README.zh-CN.md) · [图表与代码索引](docs/PAPER_INDEX.md) · [English](README.md)

</div>

RoboImagine 面向 **GPT 风格的自回归机器人世界模型**。模型接收视觉观测、动作序列和任务指令，通过**分组多样化束搜索（GDBS）**构造候选未来，并利用**语义偏好与视觉一致性反馈**进行后训练。固定锚点与分支独立的滚动历史在限制输入上下文长度的同时，保持候选生成与策略评价的条件一致。

**Hao Wu、Yuqi Li、Yizhou Zhao、Kun Wang、Xiao Luo、Yu Cheng、Yingli Tian、Laszlo A. Jeni、Ming-Hsuan Yang**
Carnegie Mellon University · The City College of New York · NTU · University of Wisconsin–Madison · UC Merced

## 结果概览

在 **RT-1 的 128 条轨迹、32 帧预测**评估中，从相同初始化出发，采用相同更新次数与评估解码器，500 步后训练相较继续 SFT 提升 **1.26 dB PSNR**，降低 **11.93% LPIPS**。训练计算成本单独统计；相同更新次数不代表相同计算量。[结果及来源 →](experiments/figures/fig08/README.zh-CN.md)

<table>
<tr><th>Ground truth</th><th>SFT initialization</th><th>Continued SFT</th><th>RoboImagine</th></tr>
<tr>
<td><img src="docs/assets/drawer-gt.png" alt="Ground-truth drawer scene"></td>
<td><img src="docs/assets/drawer-initial_sft.png" alt="SFT prediction"></td>
<td><img src="docs/assets/drawer-sft_500.png" alt="Continued-SFT prediction"></td>
<td><img src="docs/assets/drawer-hybrid_500.png" alt="RoboImagine prediction"></td>
</tr>
</table>

*图 8 的关闭抽屉案例，各图均为第 +32 帧；继续 SFT 与 RoboImagine 均增加 500 次更新。*

## 方法框架

![RoboImagine 方法框架](docs/assets/method-overview.png)

1. **监督初始化。** 冻结的视觉编解码器将场景与动态状态量化为离散 token；因果 Transformer 联合视觉—动作 token 与冻结 T5 编码的指令特征，通过 SFT 训练主干及文本交叉注意力，学习未来动态 token 的预测。
2. **分组多样化束搜索。** 在共享观测、动作和指令条件下，使用四组、每组两个 beam 构造候选。跨组 token 重叠惩罚鼓励差异，每组保留一条完整轨迹。
3. **互补反馈。** 冻结的 4B mJev（正文简称 Jev）依据任务上下文及每个视频的八帧，双向比较候选的任务进展；视觉反馈为 MAE 与 VGG-LPIPS 之和的负值。两类得分在候选集合内分别标准化，再以 0.25 和 0.75 的权重融合。
4. **保持历史一致的策略优化。** 组内相对优势调整候选 token 的概率，并向冻结的 SFT 策略正则化。评分时重放各分支实际使用的生成前缀。固定初始上下文、观测动态锚点与最近六帧状态—动作历史共同构成输入；六帧限制输入历史，不限制输出视频长度。

评估使用逐帧、宽度为四的普通束搜索，无需反馈评价器或真实未来帧。[框架图 PDF](docs/assets/method-overview.pdf) · [实际训练与奖励代码](experiments/executed_sources/rt1_hybrid/README.md)

## 快速开始

推荐使用 **Linux、Python 3.10+ 与 CUDA**。先安装相互兼容的 PyTorch 和 torchvision，然后执行：

```bash
git clone https://github.com/Alexander-wu/RoboImagine.git
cd RoboImagine
python -m pip install -e '.[notebook,evaluation]'

# 下载模型、生成真实预测视频，并计算视觉指标。
roboimagine --dataset rt1 --variant main --example-index 0 \
  --metrics --output predictions/rt1

# 使用相同样例运行对应的 SFT 初始化。
roboimagine --dataset rt1 --variant sft --example-index 0 \
  --metrics --output predictions/rt1_sft
```

首次运行会下载固定版本的模型资产和冻结 T5 编码器；LPIPS 评价还需要 VGG 权重。输出包括预测视频 `prediction.mp4`、浮点预测与 token、`memory_trace.json` 以及包含运行配置和实际计算指标的 `run.json`。

**[完整中文推理教程](docs/INFERENCE.zh-CN.md) · [英文教程](docs/INFERENCE.md) · [推理 notebook](notebooks/Inference.ipynb)**

### 已发布权重

| 数据集 | 主模型 | 对应初始化 | 演示输入 |
|---|---|---|---|
| RT-1 | Hybrid，500 步后训练 | 含文本条件的 SFT | 2 条 32 帧预测样例 |
| BridgeV2 | Hybrid，验证集选择的检查点 | 匹配的文本适配 SFT | 2 条 32 帧预测样例 |
| RoboCasa | Hybrid，500 步后训练 | 含文本条件的 SFT | 2 条 32 帧预测样例 |

六个预测器、视觉编解码器、文本适配器、动作范围和演示输入见 **[Hugging Face](https://huggingface.co/easylearning/RoboImagine)**。通过 `--dataset bridge` 或 `--dataset robocasa` 切换数据集，通过 `--variant sft` 使用对应初始化。检查点选择和评估样本范围见[推理教程](docs/INFERENCE.zh-CN.md)。

```bash
# 调整最近历史窗口。
roboimagine --dataset rt1 --window 3 --output predictions/rt1_w3

# 使用自行准备的观测、动作与指令。
roboimagine --dataset rt1 --input case.npz --horizon 32 --output predictions/custom
```

自定义输入格式、动作布局与更长序列的要求见[中文推理教程](docs/INFERENCE.zh-CN.md)。

## 实验与复现入口

| 目标 | 入口 |
|---|---|
| 从公开权重生成预测 | [模型推理](docs/INFERENCE.zh-CN.md) · [Notebook](notebooks/Inference.ipynb) |
| 阅读真实训练、奖励与评价程序 | [服务器原始实验代码](experiments/executed_sources/README.zh-CN.md) |
| 比较采样、普通 beam、GDBS 和 MCTS | [搜索消融](experiments/executed_sources/search_ablation/README.zh-CN.md) · [MCTS](experiments/executed_sources/mcts/README.zh-CN.md) |
| 核查 SFT 续训与计算开销 | [SFT 续训](experiments/executed_sources/continued_sft/README.zh-CN.md) · [图 9](experiments/figures/fig09/README.zh-CN.md) |
| 核查窗口、计时与统计聚合 | [固定 30 条轨迹的窗口实验](experiments/executed_sources/window30/README.zh-CN.md) |
| 查找每张图表的具体实现 | [图表索引](docs/PAPER_INDEX.md) |
| 使用已归档实验输出重绘图表 | [中文实验教程](experiments/README.zh-CN.md) · [Run All notebook](experiments/Reproduce_All.ipynb) |

`executed_sources` 保存服务器实际执行的训练、搜索、奖励、视频推理、指标计算与聚合代码。中英文指南说明实验配置、数据准备与迁移环境时需要替换的路径。图表 notebook 则用于已有实验输出的可视化。

### 四个研究问题

- **RQ1：总体效果。** 三个数据集的视觉质量、任务语义、多模型评价及定性比较。
- **RQ2：后训练收益。** 同初始化下的 SFT 续训对照、检查点曲线与训练 GPU 小时。
- **RQ3：组件作用。** 语义、视觉与混合反馈，以及采样、普通 beam、GDBS、似然引导 MCTS；Qwen 消融统一使用 97 条配对有效轨迹。
- **RQ4：长序列与效率。** 有界历史的质量与计算开销。表 5 / 图 14 使用相同 30 条长序列合格测试轨迹，覆盖五个预测长度；图 13 为独立的历史 11 条轨迹实验。

各实验保留各自的样本定义。表 2 的 BridgeV2 结果使用明确标注的 32 条验证轨迹；窗口实验的 30 条轨迹与主结果的 128 条轨迹不能混为同一评估集合。

## 目录与发布说明

- `src/roboimagine/`：模型、推理、搜索、记忆管理与指标计算。
- `experiments/executed_sources/`：实际训练和评价代码。
- `experiments/figures/`、`experiments/tables/`：实验图 6–14 与表 1–5 的来源、代码和指南。
- `provenance/`：权重与代码发布清单及验证记录。
- `docs/`：推理指南、图表索引与 README 使用的论文图像。

已公开主模型及 SFT 权重和演示样例；完整预处理评估集合、评价器资产及全部中间/消融权重未全部打包，完整运行所需输入在[原始程序指南](experiments/executed_sources/README.zh-CN.md)中说明。验证记录见 [model validation](provenance/model_validation.json) 与 [download validation](provenance/public_download_validation.json)，第三方素材及代码条款见 [NOTICE](NOTICE.md)。

实现或复现问题欢迎通过 [GitHub Issues](https://github.com/Alexander-wu/RoboImagine/issues) 交流。
