# Figure 6

RT-1 case032 + BridgeV2 case079（当前验证所选检查点）；真实帧8/16/24/32和固定ROI。保留必要Baseline图片，不含Baseline模型代码。

From `experiments/`: `python run_all.py --figures 6`.

Entry: `build_figure.py`. Inputs are JSON/CSV plus local image/font assets. The current manuscript caption is in `paper_caption.tex`. Outputs are written under `reproduced/figures/fig06`; no model inference is performed.
