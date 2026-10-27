# 图 9 — 训练 GPU 小时与质量

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

GPU 小时为分配 GPU 数乘训练循环墙钟时间：归档的 500 步 SFT 续训为 1.063，GDBS 为 28.152；分别使用 4 张和 8 张 H20。统计包括循环内搜索、反馈、同步及保存检查点，不含初始化、预处理、最终评测和之前的 50k 步 SFT。这是单次训练成本，不是推理延迟或等算力比较。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 9
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig09/figure.pdf`. 同目录保存运行日志。

## 输入和代码

- [data.json](data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
