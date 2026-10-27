# 图 6 — GT 与模型预测的四时刻对比

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

RT-1 case032 与 BridgeV2 case079 展示第 8、16、24、32 帧，案例内采用固定裁剪区域。Bridge 使用当前验证集所选检查点。这是选定的定性案例，不是均值实验；保留对比所需的基线图片，不包含基线模型实现。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 6
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig06/baseline_comparison.pdf`. 同目录保存运行日志。

## 输入和代码

- [selection.json](selection.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
