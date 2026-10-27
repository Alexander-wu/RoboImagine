# 图 10 — 三数据集 SFT 对照

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

RT-1 案例使用 visual-only 的 500 步版本，BridgeV2 和 RoboCasa 使用 combined-feedback 的 500 步版本。此处 Bridge 是历史 500 步权重，区别于当前公开的验证集所选 250 步权重以及图 6，不能混为同一检查点。素材来源见 selection 清单。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 10
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig10/r2_three_dataset_comparison.pdf`. 同目录保存运行日志。

## 输入和代码

- [selection.json](selection.json)
- [render_figure.py](render_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
