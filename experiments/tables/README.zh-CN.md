# 论文表格复现

[English](README.md) | [简体中文](README.zh-CN.md) | [实验总指南](../README.zh-CN.md)

## 运行方法

安装实验依赖后，在仓库根目录运行：

```bash
python experiments/tables/build_tables.py
```

[构建脚本](build_tables.py) 将五张表编译为五页独立预览，输出 `experiments/reproduced/tables/tables.pdf`，同时保留执行回执和日志。需要 TeX 与 `latexmk`；没有编译器时记录跳过。[总 notebook](../Reproduce_All.ipynb) 会在图形重绘后调用此脚本。

## 对应论文内容

| 表 | 来源 | 说明 |
|---|---|---|
| [1](table01.tex) | [主表 CSV](data/our_model_main_tables.csv) | 作者提供的 RoboAlign-Judge 均值和偏差，不从原始评分响应重新估计 |
| [2](table02.tex) | [主表 CSV](data/our_model_main_tables.csv) | 我们模型的三数据集指标；RT-1/RoboCasa 为 128 条报告轨迹，Bridge 为 32 条验证轨迹及验证集所选检查点 |
| [3](table03.tex) | [视觉指标](data/table03_visual_results.json)、[Qwen](data/table03_qwen97_results.json) | 反馈/搜索消融：视觉指标 128 条，六个配置的 Qwen 使用同 97 条有效配对轨迹 |
| [4](table04.tex) | [搜索结果](data/table04_search_results.json) | 随机采样/MCTS/普通 beam/GDBS 最终 500 步版本与 SFT，在完整 128 条轨迹上对比，附每次轨迹呈现的训练 GPU 秒 |
| [5](table05.tex) | [同 30 条结果](data/table05_matched30_results.json) | 固定权重，保留锚点的 W=1/3/6/12，在同 30 条 test 轨迹上评估 H=16/32/48/64/96 |

表 1–2 只导出我们模型的行，不保留外部基线排名；表 3–5 保留必要对照。TeX 保存论文格式化数值，JSON/CSV 保留来源汇总与协议。此命令重新编译排版，不进行新训练、推理或 VLM 评分。

## 消融解读要点

- 表 3(A) 的 Hybrid 和 3(B) 的 GDBS 是**同一配置**，视觉指标和当前配对 Qwen 分数一致。旧 9 条搜索试验不是当前表格；Qwen JSON 保留随机采样配置三条响应的有上限格式修复记录。
- 独立 Qwen 评测不同于训练 Jev 反馈，不同指标的排序不一定相同。
- 表 4 成本为 GPU 数乘训练循环墙钟时间，再除以更新次数乘全局 batch；不是每个独立视频的推理耗时，也不是等算力比较。MCTS 的具体设定见协议文件。
- 表 5 质量先对帧平均再对轨迹平均；每条轨迹的三次计时先平均，再对 30 条轨迹平均。来源保留包含仅当前帧的 750 条质量和 2,250 条计时记录，表格展示其中四个带锚点窗口；另保留单检查点下的配对 bootstrap 汇总。
- 本组 W6/H32 的 22.20 dB 与主表原 128 条的 22.57 dB 来自不同队列，不能直接比较；图 13 是另一个历史 11 条轨迹的实验。

完整设置见[消融协议](../README.zh-CN.md#消融实验改变什么保持什么)，从权重运行的支持范围见[推理教程](../../docs/INFERENCE.zh-CN.md)。
