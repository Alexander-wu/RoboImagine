# RoboImagine project page

A dependency-free, responsive research project page. It includes synchronized ground-truth/prediction playback for three released examples, interactive checkpoint curves, a qualitative case, and links to the actual experiment implementations.

## Local preview

From the repository root:

```bash
python -m http.server 8090 --directory website
```

Open http://localhost:8090. A server is required because the page loads `data.json`.

## Sources

- Method overview: the current manuscript’s framework figure, rendered from its vector PDF. Explanatory text follows Sections 3.1–3.4. Only the finished PNG/PDF are included; method-figure drawing source is not distributed.

- Demonstrations: example 0 and its saved main-model prediction for each dataset from `easylearning/RoboImagine@cb4f5e2bdfbbbff2ad6caffa058d50fa21ec3ec7`. The WebP atlases encode all 32 frames in an 8-column × 4-row grid. No interpolation is used. Source NPZ SHA-256 hashes are recorded in `data.json`.
- Learning curves: `experiments/figures/fig08/curve_data.json`. The page reads measured checkpoints, using straight line segments. It performs no inference or metric evaluation.
- Drawer case: `experiments/figures/fig08/assets/case_034/`, all at frame +32. The shared references and two 500-update checkpoints are labeled explicitly.
- Typography: EB Garamond; OFL notice is retained in `assets/`.
- Dataset frames retain their original dataset terms. See the repository `NOTICE.md`.

The page is static and portable: copy this directory to any static host. It contains no API keys, analytics, private manuscript link, model-diagram source code, or backend dependencies. It does not change any training or evaluation implementation.

## 中文

这是独立的静态项目主页，支持三个公开样例的真实视频与预测视频同步播放，以及训练检查点曲线的指标切换。运行上面的命令后，在浏览器打开 `http://localhost:8090` 即可预览。

示例帧来自已发布权重对应的归档预测，曲线来自实际实验记录。网页只展示结果，不重新训练或计算指标。主页源码可直接部署至静态网站服务；论文方法示意图的绘图代码不在此目录中。
