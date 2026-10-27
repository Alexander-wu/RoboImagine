# 服务器已执行实验源码

[English](README.md) | [项目主页](../../README.md) | [已有结果绘图工具](../README.zh-CN.md)

这里整理的是**服务器上实际运行过的训练、推理、指标计算、奖励评分和汇总代码**，从已有实验目录只读取回。不是把论文里的数字再打印或画一遍。本次整理没有重新训练、生成预测、调用评测模型或重算指标。

## 先看真实计算代码

- **从预测像素和 GT 计算视觉指标：**[原始 Metrics 与评估循环](rt1_hybrid/code/src/futureworlds/evaluation.py)。逐帧计算 PSNR、SSIM、VGG-LPIPS、MAE，再由实验 worker 做帧平均及轨迹平均。
- **搜索候选未来：**[GDBS](rt1_hybrid/code/src/futureworlds/search.py)、[普通 beam](search_ablation/code/src/futureworlds/ordinary_search.py)、[采样与滚动历史](search_ablation/code/src/futureworlds/memory.py)、[实际 MCTS](mcts/source_v1/mcts_search.py)。
- **后训练：**[策略优化](rt1_hybrid/code/src/futureworlds/training.py)、[候选生成—反馈—更新循环](rt1_hybrid/train.py)、[视觉奖励及反馈组合](rt1_hybrid/common.py)、[成对 judge 服务](rt1_hybrid/judge_service.py)。
- **SFT 续训：**[最终四卡 train.py](continued_sft/source_sft_v4/sft/train.py)、[批量与初始化定义](continued_sft/source_sft_v4/sft/sft_common.py)。旧八卡 protocol 是缓存准备记录；实际完成的续训由 v4 和迁移回执共同定义。
- **独立 Qwen 六维评分：**[提示词与解析器](qwen97/source/original_common.py)、[视频评分及格式修复重试](qwen97/source/worker.py)、[共同样本汇总](qwen97/source/aggregate.py)。
- **窗口质量和效率：**[解码计时](window30/source/profile_decoder.py)、[生成及像素评分](window30/source/worker.py)、[完整网格、重复计时和 bootstrap 汇总](window30/source/aggregate.py)。

## 与论文对应

| 论文位置 | 实验 | 实际代码 | 配置 |
|---|---|---|---|
| Table 2/3/4; Figures 8/9/10 | [RT-1 Hybrid / GDBS](rt1_hybrid/README.zh-CN.md) | [执行](rt1_hybrid/train.py) · [具体实现／评分](rt1_hybrid/common.py) | [配置](rt1_hybrid/protocol.json) |
| Table 3(A); Figure 10 | [RT-1 visual-only](rt1_visual/README.zh-CN.md) | [执行](rt1_visual/train.py) · [具体实现／评分](rt1_visual/common.py) | [配置](rt1_visual/protocol.json) |
| Table 3(A) | [RT-1 semantic-only](rt1_semantic/README.zh-CN.md) | [执行](rt1_semantic/train.py) · [具体实现／评分](rt1_semantic/judge_service.py) | [配置](rt1_semantic/protocol.json) |
| Table 3(B)/4; Figure 8 | [Sampling / ordinary beam](search_ablation/README.zh-CN.md) | [执行](search_ablation/train.py) · [具体实现／评分](search_ablation/common.py) | [配置](search_ablation/search_ordinary/protocol.json) |
| Table 2; Figure 10 | [RoboCasa / earlier Bridge suite](dataset_suite/README.zh-CN.md) | [执行](dataset_suite/train.py) · [具体实现／评分](dataset_suite/evaluate.py) | [配置](dataset_suite/robocasa_hybrid500/protocol.json) |
| Table 2; Figure 6 | [Bridge selected checkpoint](bridge_selected/README.zh-CN.md) | [执行](bridge_selected/source_v1/train_retry.py) · [具体实现／评分](bridge_selected/source_v1/evaluate_retry.py) | [配置](bridge_selected/configs/lr2e7/protocol.json) |
| Table 4; Figures 8/11 | [Likelihood-guided MCTS](mcts/README.zh-CN.md) | [执行](mcts/source_v1/train_mcts.py) · [具体实现／评分](mcts/source_v1/mcts_search.py) | [配置](mcts/config_v1/protocol.json) |
| Figures 8/9 | [Continued SFT](continued_sft/README.zh-CN.md) | [执行](continued_sft/source_sft_v4/sft/train.py) · [具体实现／评分](continued_sft/source_sft_v4/sft/evaluate.py) | [配置](continued_sft/sft/rt1_sft500_v1/four_gpu_preparation_admission.json) |
| Figure 8 | [SFT / Hybrid intermediate checkpoints](learning_curves/README.zh-CN.md) | [执行](learning_curves/source_v1/evaluator.py) · [具体实现／评分](learning_curves/source_v1/evaluator.py) | [配置](learning_curves/hybrid_000250/protocol.json) |
| Figure 8 | [MCTS / ordinary intermediate checkpoints](search_curves/README.zh-CN.md) | [执行](search_curves/v1/source/worker.py) · [具体实现／评分](search_curves/v1/source/worker.py) | [配置](search_curves/v1/rank0/protocol.json) |
| Table 3; Figure 7 (Qwen) | [Independent six-dimensional scoring](qwen97/README.zh-CN.md) | [执行](qwen97/source/worker.py) · [具体实现／评分](qwen97/source/aggregate.py) | [配置](qwen97/source/cohort.json) |
| Figure 11 | [Between-model feedback](prediction_feedback/README.zh-CN.md) | [执行](prediction_feedback/score_controller_v1.py) · [具体实现／评分](prediction_feedback/scoring_v1/judge_service_h32.py) | [配置](prediction_feedback/scoring_v1/protocol.json) |
| Figure 13 | [Historical long-horizon memory](long_memory/README.zh-CN.md) | [执行](long_memory/code_v2/worker.py) · [具体实现／评分](long_memory/code_v2/aggregate.py) | [配置](long_memory/eval_v2/protocol.json) |
| Table 5; Figure 14 | [Matched-30 window / efficiency](window30/README.zh-CN.md) | [执行](window30/source/worker.py) · [具体实现／评分](window30/source/aggregate.py) | [配置](window30/protocol.json) |

图 12 保留在[搜索案例档案](../figures/fig12/README.zh-CN.md)，其 ancestry 来自上面的 GDBS 实现。表 1 的 RoboAlign-Judge 作者提供汇总、图 7 的豆包和对话模型评分，并不会因为导出了数字就变成已经提供原始评价程序；本目录不为缺失部分编造代码。

## 运行前需要哪些文件

1. **输入数据与缓存。**[RT-1 原始准备程序](rt1_semantic/prepare.py)和[Bridge/RoboCasa 准备程序](dataset_suite/prepare.py)读取真实图像、动作及指令，编码初始上下文、动态锚点、动作和文本特征。对应训练实验使用 5200 个训练窗口、8 个 smoke 窗口、128 个报告窗口。必须按各自协议使用真实 ID 和划分；历史目录里的 `dev` 名字不能用于把验证集改称测试集。
2. **初始化和训练。**`common.py` 加载 codec、SFT 权重、文本适配器与冻结 T5；`train.py` 实际构造候选、计算反馈、更新参数、保存权重及回执。已有预测只需评分时，不必重新训练。
3. **预测。**对应评估 worker 加载指定权重，从观测前缀、动作和指令生成视频，保存浮点像素及 token。评估时 beam=4 与后训练期间候选搜索是两件事。
4. **评分。**视觉指标输入是预测像素和 GT，不是论文均值或截图；语义评分输入是视频与任务条件，并保留真实输出、解析结果与重试记录。
5. **汇总。**使用原始 aggregate，先核对样本 ID、配置网格和文件来源再平均。固定 30 条窗口实验共有 750 条质量记录、2250 条计时记录；3 次重复不是 90 条独立轨迹。它与主表 128 条、历史长序列 11 条分别统计。
6. **最后绘图。**原有绘图 notebook 只负责从已保存结果生成图表，不等于重做模型实验。

### 已开放权重的真实推理入口

现有入口可以加载公开权重，生成预测并从像素计算指标：

```bash
python -m pip install -e '.[evaluation,notebook]'
roboimagine --dataset rt1 --variant main --example-index 0 --metrics --output predictions/rt1_main
roboimagine --dataset rt1 --variant sft --example-index 0 --metrics --output predictions/rt1_sft
```

这是样例级真实计算，不是完整论文队列的复现。见[推理教程](../../docs/INFERENCE.zh-CN.md)与[现有推理 notebook](../../notebooks/Inference.ipynb)。本次整理没有执行这些 GPU 命令。

### 历史实验入口与顺序

以下是**恢复输入目录、依赖、协议和来源门禁之后**的原始入口形式。仅克隆仓库还不足以运行完整队列；不要把旧调度脚本提交到现用集群。

| 实验 | 入口形式 | 运行条件 |
|---|---|---|
| RT-1 Hybrid | `torchrun --standalone --nproc_per_node=8 worker_entry.py train` | `rt1_hybrid` 目录，8 个可见 CUDA 设备，准备缓存与 gate |
| 采样／普通 beam | `torchrun --standalone --nproc_per_node=8 worker_entry.py train` | `search_ablation` 目录，`R2_RUN_DIR` 选择 `search_sample` 或 `search_ordinary` |
| MCTS | `torchrun --standalone --nproc_per_node=4 worker_entry.py train_mcts` | `mcts/source_v1`，`R2_RUN_DIR` 和新的 `R2_OUTPUT_ROOT`；resume 须有原检查点回执 |
| 续训 SFT | `torchrun --standalone --nproc_per_node=4 worker_entry.py train` | `continued_sft/source_sft_v4/sft`，`R2_SFT_OUTPUT`、准备缓存及迁移记录 |
| Bridge 评估 | `torchrun --standalone --nproc_per_node=4 worker_entry.py evaluate_retry` | `bridge_selected/source_v1`，`R2_RUN_DIR`、`BRIDGE_EVAL_SPEC` 指定样本、权重哈希、输出 |
| Qwen | `python worker.py`，全部完成后 `python aggregate.py` | `qwen97/source`，每 worker 一卡、独立 `EVALUATION_RANK`、已准备队列和固定评价器 |
| 固定 30 条窗口实验 | `python worker.py --rank 0`，之后 `python worker.py --rank 0 --score` | `window30/source`，rank 0–3 各一卡，固定输入和协议；全部完成再 aggregate |
| 历史长视频 | `python worker.py --rank 0 --shards 4` | `long_memory/code_v2`，rank 0–3，各队列使用原始 manifest |

保留 `futureworlds` Python 命名空间，是因为原代码如此导入，避免整理时改变执行逻辑；这只是内部包名。各组保留自己的运行时源码快照，不要因同名就擅自替换为另一组的版本。

## 来源校验与迁移边界

机器路径替换为 `/opt/roboimagine-workspace`、`/opt/roboimagine-scratch`，GPU UUID 使用 `GPU-REPLACE-UUID-*`。它们是占位符，不是下载地址。算法语句、数值超参数、采样与平均规则、评价提示词保持原样；打包时用 AST 比较核对 Python 只发生上述字符串替换。

[来源清单](SOURCE_MANIFEST.json)逐文件保存原始 SHA256 和发布 SHA256。**代码内部的源码／输入／权重哈希门禁仍对应历史实验。**迁移后可能因路径或缺少回执而拒绝运行；这里没有关闭门禁，也没有伪造成功回执。运行迁移版需要在新目录记录本地协议、文件来源及哈希，并保留原始记录用于追溯。

[HF 已发布内容](https://huggingface.co/easylearning/RoboImagine)包括主模型、SFT 和演示输入；完整训练／评估准备数据、全部消融与中间权重、缓存／回执目录、评价器模型尚未全部随包发布。因此，**源码整理不表示所有论文实验已能从空环境一键运行**。缺失项不使用猜测链接或预填结果替代。已取消的 Bridge 第二学习率组和与实验算法无关的集群调度脚本不纳入入口。

历史源码使用 Python 3.11+（含 `hashlib.file_digest`），以及兼容的 CUDA/PyTorch、[模型依赖](../../pyproject.toml)和 LPIPS/PIQA 依赖；评价器按原配置准备版本与本地文件。本包不声称已提供完整历史环境锁文件。

仅验证归档文件，不启动实验：

```bash
python experiments/executed_sources/verify_snapshot.py
```

这条命令只检查哈希和 Python 语法，不是实验复现。图 1–5 的模型示意图绘图代码不在发布范围内。
