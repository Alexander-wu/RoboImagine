# 图 8 — SFT 续训与搜索变体训练曲线

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

SFT 续训、GDBS、普通 beam、MCTS 在同一批 128 条 RT-1 报告轨迹上评估 100/250/500 步检查点，预测 32 帧并使用相同评估解码器。定性部分使用关闭抽屉 case034 的第 32 个预测帧。相同更新步数不等于相同计算成本；公开的主模型/SFT 权重包不包含这些中间检查点。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 8
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig08/figure.pdf`. 同目录保存运行日志。

## 输入和代码

- [curve_data.json](curve_data.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
