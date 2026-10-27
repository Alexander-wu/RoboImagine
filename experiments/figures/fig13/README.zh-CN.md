# 图 13 — 64／128帧记忆对照

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

图中历史实验使用同 11 条 held-out 轨迹，分别统计前 64 帧和全部 128 帧。完整记忆为固定动态锚点加最近 6 个状态；仅当前帧配置移除动态锚点、只保留最近 1 个状态；两者均保留初始上下文。该对比不能独立证明锚点贡献。数据文件还保留历史开发集记录，但绘图选择 11 条 held-out 轨迹，不可标为新的 30 条实验。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 13
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig13/figure.pdf`. 同目录保存运行日志。

## 输入和代码

- [data.json](data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
