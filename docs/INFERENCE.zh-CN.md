# 从公开权重运行 RoboImagine

[English](INFERENCE.md) | [简体中文](INFERENCE.zh-CN.md) | [项目主页](../README.md)

本发布包含实际 AR 模型、训练后的文本适配器、FSQ 视觉 codec、动作范围、固定锚点滚动记忆、普通 beam 和分组多样化 beam。提供 RT-1、BridgeV2、RoboCasa 的主模型和对应 SFT 初始化。结构示意图绘图脚本不属于本发布范围。

## 安装

推荐 Linux CUDA、Python 3.10+ 及相互兼容的 PyTorch/torchvision。加载器支持 CPU，但推理较慢；发布验证在 NVIDIA H20 上使用 FP32。先安装适合本机 CUDA 的 PyTorch，再执行：

```bash
git clone https://github.com/Alexander-wu/RoboImagine.git
cd RoboImagine
python -m pip install -e '.[notebook]'
```

固定 Transformers 4.57.1 和 diffusers 0.35.2。不同内核或库版本可能影响数值一致性，实际验证见 [model_validation.json](../provenance/model_validation.json)。

## 一条命令生成视频

```bash
roboimagine --dataset rt1 --variant main --example-index 0 --output predictions/rt1
roboimagine --dataset bridge --variant main --example-index 0 --output predictions/bridge
roboimagine --dataset robocasa --variant main --example-index 0 --output predictions/robocasa
```

首次运行从 [Hugging Face](https://huggingface.co/easylearning/RoboImagine) 下载所选模型、适配器、codec、动作范围及案例，同时下载配置中固定版本的 `google-t5/t5-base`。不调用评价器 API。每个数据集/版本的模型及 codec 约 1 GB，另有 T5 和样例；六套检查点发布约 4.9 GB（不含 T5）。`--variant sft` 选择对应 SFT 初始化，共用 codec。

输出包括：

- `prediction.mp4`：预测视频；4 fps 仅为观看设置，不代表传感器或控制频率。
- `ground_truth.mp4`：输入含真实未来时生成参考视频。
- `prediction.npz`：原始 float32 预测像素及离散 token。
- `memory_trace.json`：逐步实际保留的历史状态索引。
- `run.json`：参数、可计算时的 PSNR/MAE、耗时及可选归档结果对照。单次未预热演示耗时不是论文效率实验。

SSIM 和 VGG-LPIPS 可通过 `pip install -e '.[evaluation]'` 安装，并添加 `--metrics`；首次使用 torchvision 可能下载 VGG 权重。指标先在预测帧上平均。每个数据集的两个演示案例不能替代完整评估均值。

## Jupyter Run All

打开 [Inference.ipynb](../notebooks/Inference.ipynb)，选择数据集，在安装好依赖的内核中运行全部单元。Notebook 下载所需文件、生成并显示视频，并与保存的主模型预测进行校验。参数区也支持已有本地 bundle、输入、参考结果和 T5 路径，便于离线使用。

## 自己的输入

```python
np.savez_compressed('case.npz',
    observations=observations,  # uint8, [2,256,320,3], RGB
    actions=actions,            # float32, [H+1,A]，原数据集动作布局
    instruction=np.array(instruction))
```

运行 `roboimagine --dataset rt1 --input case.npz --horizon 32`。可选 `ground_truth` 为 uint8 `[H,256,320,3]`，仅在预测完成后用于比较。模型接收两个初始观测、计划动作和指令，不接收未来 RGB。RT-1/Bridge 动作维数为 13，RoboCasa 为 52，后者对应归档的四步动作布局。换用其他机器人动作时必须匹配预处理、单位与布局。

附带样例支持预测 32 帧；更长视频需要对应长度的真实计划动作，不自动重复或填充动作。`--window` 可设为 1、3、6、12，改变最近动态状态数，同时保留初始场景及固定动态锚点；`--no-anchor --window 1` 使用仅当前状态基线。输出长度不受输入历史窗口大小直接限制。

## 权重与复现范围

- RT-1/RoboCasa 主权重为 500 步后训练；Bridge 主权重为验证集选择的 250 步版本，与较早的 500 步可视化不同。
- Bridge 的带文本 SFT 权重是重新训练实际使用的初始化，不是名称相似的旧无文本权重。
- 主表 Bridge 指标来自明确标注的 32 条验证轨迹；两个公开演示案例按原评估索引选自另一个报告队列，不声称能复现主表均值。
- 提供搜索实现与策略目标基础组件，但不包含完整训练启动器、奖励服务托管或完整原始训练数据。没有把占位命令当作完整训练流程。
- 记忆实验、完整 128 条搜索评估及较小评价器队列分别保留各自定义，见[实验教程](../experiments/README.zh-CN.md)。

## 发布验证

在测试 CUDA 环境（PyTorch 2.10.0、Transformers 4.57.1、diffusers 0.35.2）下，六个主模型案例的 token 与归档完全一致，float 像素最大误差为零。三套对应 SFT 各有一个形状和有限值检查通过。推理 notebook 使用已校验的 RT-1 缓存及本地固定版本 T5 完成 Run All（含 MP4）。这是对演示案例的验证，不是所有轨迹的验证。

权重固定版本：`easylearning/RoboImagine@cb4f5e2bdfbbbff2ad6caffa058d50fa21ec3ec7`。

另在全新缓存中，从公开 HF 地址下载完整 RT-1 主模型 bundle 及输入/参考结果并验证文件。公开下载测试和 CUDA notebook 推理测试分别记录在 [provenance/](../provenance/) 中。
