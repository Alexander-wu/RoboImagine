# 图 7 — 多模型评测与输出摘录

**实际实验计算代码：** [qwen97](../../executed_sources/qwen97/README.zh-CN.md)
本页 notebook 仅重绘保存结果；训练、生成和评分见上面的原始程序。

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

Qwen、豆包、GPT-6 Astra 分别使用 97、10、50 条方法间配对轨迹，三个评价器并非使用同一批样本。GPT-6 Astra 表示当时配置的会话模型评阅，不是固定版本的 API 评测。右侧文本是实际单条输出，与左侧均值分开；重绘不会调用评价 API。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 7
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig07/figure.pdf`. 同目录保存运行日志。

## 输入和代码

- [data.json](data.json)
- [excerpts.json](excerpts.json)
- [session_model_provenance.json](session_model_provenance.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
