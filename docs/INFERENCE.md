# Run RoboImagine from released weights

[English](INFERENCE.md) | [简体中文](INFERENCE.zh-CN.md) | [Project home](../README.md)

This release includes the actual AR model, trained text adapters, FSQ visual codec, action ranges, fixed-anchor rolling memory, ordinary beam and grouped diverse beam search. It contains inference checkpoints for RT-1, BridgeV2 and RoboCasa, plus their SFT initializations. Architecture illustration scripts are not included.

## Installation

Use a Linux CUDA environment with Python 3.10+ and a compatible PyTorch/torchvision pair. CPU is supported by the loader but slow; the release checks use FP32 on an NVIDIA H20. Install PyTorch for your CUDA environment first, then:

```bash
git clone https://github.com/Alexander-wu/RoboImagine.git
cd RoboImagine
python -m pip install -e '.[notebook]'
```

Transformers 4.57.1 and diffusers 0.35.2 are pinned. Numerical parity may change with different kernels or library versions. `provenance/model_validation.json` records the actual validation results.

## One command

```bash
roboimagine --dataset rt1 --variant main --example-index 0 --output predictions/rt1
roboimagine --dataset bridge --variant main --example-index 0 --output predictions/bridge
roboimagine --dataset robocasa --variant main --example-index 0 --output predictions/robocasa
```

The first run downloads the selected dataset/variant's model, adapters, codec, action ranges and example from [Hugging Face](https://huggingface.co/easylearning/RoboImagine). It also downloads the frozen `google-t5/t5-base` encoder at the pinned revision recorded in the configuration. It does not call any evaluator API. Expect roughly 1 GB of model/codec assets per dataset/variant plus T5 and sample files; the whole six-checkpoint release is about 4.9 GB before T5. Use `--variant sft` for the matching SFT initialization; the codec is shared.

Outputs:

- `prediction.mp4`: generated video (4 fps is a viewing convention, not a claimed sensor/control frequency).
- `ground_truth.mp4`: reference future, if supplied in the example.
- `prediction.npz`: original float32 prediction pixels and discrete tokens.
- `memory_trace.json`: the actual retained state indices at each autoregressive step.
- `run.json`: parameters, PSNR/MAE where GT exists, elapsed time and optional archived-output parity check. A single unwarmed demo timing is not the paper's efficiency benchmark.

For SSIM and VGG-LPIPS, install `pip install -e '.[evaluation]'` and add `--metrics`; torchvision may download VGG weights on first use. Metrics first average over predicted frames. Do not treat the included two examples as a replacement for the full evaluation cohort.

## Notebook

Open [`notebooks/Inference.ipynb`](../notebooks/Inference.ipynb), choose the dataset, then Run All. Dependencies should be installed in the notebook's active kernel. It downloads the requested files, generates the video, displays it, and checks against the saved main-checkpoint prediction.

## Your own inputs

```python
np.savez_compressed('case.npz',
    observations=observations,  # uint8, [2,256,320,3], RGB
    actions=actions,            # float32, [H+1,A], native dataset layout
    instruction=np.array(instruction))
```

Use `roboimagine --dataset rt1 --input case.npz --horizon 32`. Optional `ground_truth` is uint8 `[H,256,320,3]` and is used only after prediction for comparison. The model receives exactly two initial observations, planned actions and the instruction, never future RGB. Action dimensions are 13 for RT-1/Bridge and 52 for RoboCasa; RoboCasa's 52 values represent the archived four-step action layout. Arbitrary actions from another robot cannot be substituted without matching the preprocessing and units.

The bundled demo clips support 32 predicted frames. Longer rollouts require the corresponding number of real planned action steps. No action repetition or padding is performed. `--window 1/3/6/12` changes the number of recent dynamic states; the fixed initial scene and dynamic anchor remain. `--no-anchor --window 1` runs the current-state baseline. Output length is independent of the bounded input history.

## Scope

- RT-1/RoboCasa main checkpoints use 500 post-training updates. Bridge main is the validation-selected 250-update checkpoint, distinct from the earlier 500-update visualizations.
- Bridge's text-enabled SFT initialization is the actual initialization used by its rerun, not the older no-text checkpoint with a similar directory name.
- Main-table Bridge scores use the documented 32-trajectory validation cohort. These two released examples are drawn by original evaluation index from the separate reporting pool. They are not claimed to reproduce the main-table average.
- Search implementations and policy-objective primitives are included. A complete training launcher, reward-service hosting and full raw training datasets are not part of this inference release. No placeholder training entrypoint is presented as a working reproduction command.
- The memory experiments, full-128 search evaluation and smaller judge subsets retain their original cohort definitions in `experiments/README.md`.

## Release verification

The six main-model examples reproduce their archived token sequences exactly and have zero maximum float-pixel difference under the tested CUDA environment (PyTorch 2.10.0, Transformers 4.57.1, diffusers 0.35.2). One example from each matching SFT checkpoint also passes shape and finite-output checks. The inference notebook passes Run All using the verified cached RT-1 bundle and pinned local T5, including MP4 output. See `provenance/model_validation.json`. This checks these examples, not all dataset trajectories.

Pinned checkpoint release: `easylearning/RoboImagine@cb4f5e2bdfbbbff2ad6caffa058d50fa21ec3ec7`.

The released loader also passes a fresh-cache public download of the complete RT-1 main bundle and example/reference pair, with all downloaded files verified. The download test and CUDA notebook inference test are recorded separately in `provenance/`.
