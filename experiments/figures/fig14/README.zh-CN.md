# 图 14 — 窗口质量与推理效率

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

固定 Hybrid500 权重，在相同 30 条满足长序列要求的 RT-1 test 轨迹上评估 H=16/32/48/64/96。图中展示保留锚点的 W=1/3/6/12；原始结果还保留仅当前帧配置。质量先对帧平均再对轨迹平均；三次计时先在轨迹内平均。预热后的生成时间不含模型加载、文本缓存构建、评分及文件 I/O；峰值显存为含常驻模型的 PyTorch allocated memory，不是 reserved memory。本组 W6/H32 为 22.20 dB，主表原 128 条为 22.57 dB，两个均值不能直接比较。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 14
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig14/figure14_heatmap_bars.pdf`. 同目录保存运行日志。

## 输入和代码

- [data.json](data.json)
- [plotted_data.json](plotted_data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
