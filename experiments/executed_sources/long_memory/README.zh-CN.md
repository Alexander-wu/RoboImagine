# Historical long-horizon memory

[English](README.md) | [实验源码总索引](../README.zh-CN.md)

对应：Figure 13。图中使用 heldout_test128（11 条）；其他诊断队列分开保留。

- [原始执行代码](code_v2/worker.py)
- [具体实现／评分代码](code_v2/aggregate.py)
- [原始配置或队列说明](eval_v2/protocol.json)

这些文件从已运行的服务器目录读取。没有重新训练、推理或生成评分。计算逻辑未改；机器目录和 GPU UUID 改为占位符。每个文件的原始及发布 SHA256 见 [来源清单](../SOURCE_MANIFEST.json)。

输入、输出及运行顺序见[总教程](../README.zh-CN.md#运行前需要哪些文件)。完整准备数据、部分消融权重及历史回执尚未包含，不能直接把本目录当成从空环境运行全部实验的入口。历史哈希门禁保留；迁移时应在新目录记录本地来源，不能关闭检查或覆盖旧记录。
