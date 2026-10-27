# 图 12 — 真实 GDBS 搜索案例

[English](README.md) | [简体中文](README.zh-CN.md)

## 对应论文与实验设置

从 8 个训练条件中，按语义评分离散程度的增幅选取一个实际案例。保留分支的祖先关系及反馈来自真实记录，不是人为构造的树。该图是选定案例，不是测试集均值；可通过 raw_traces.json 和 expected_ancestry.json 追溯分支。

## 复现

先按[实验总指南](../../README.zh-CN.md)安装依赖。在仓库根目录运行：

```bash
python experiments/run_all.py --figures 12
```

或打开[Reproduce.ipynb](Reproduce.ipynb)，选择已安装依赖的内核并运行全部单元。

生成的 PDF：`experiments/reproduced/figures/fig12/gdbs_case_study.pdf`. 同目录保存运行日志。

## 输入和代码

- [raw_traces.json](raw_traces.json)
- [expected_ancestry.json](expected_ancestry.json)
- [asset_manifest.json](asset_manifest.json)
- [build_figure.py](build_figure.py)
- [paper_caption.tex](paper_caption.tex)

该入口基于已保存结果重绘，不重新训练、生成整套预测或调用付费评价器。基于公开权重生成新预测，请使用[推理教程](../../../docs/INFERENCE.zh-CN.md)。
