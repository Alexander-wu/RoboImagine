# 图 11 — 完整128条预测距离与反馈

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

使用完整 128 条配对 RT-1 报告轨迹，对比 SFT、MCTS、普通 beam 与 GDBS 的最终预测。距离是模型间预测距离，不是一次搜索内部的候选多样性；更大距离不等于更高质量。语义反馈来自冻结的训练评价器 Jev，区别于表 3 的独立 Qwen 打分；两类反馈均为越大越好。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 11
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig11/figure.pdf`. 同目录保存运行日志。

## 输入和代码

- [results.json](results.json)
- [per_case.csv](per_case.csv)
- [distance_pairs.csv](distance_pairs.csv)
- [quality_reference.json](quality_reference.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
