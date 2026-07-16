# RGBD-repair

[中文](README.md) | [English](README_EN.md)

基于 LingBot-Depth 的 RGB-D 深度图修复实验记录，包含最终 calibrated local fusion 方案、三个离线 Demo，以及机器人 Orin 上的 TensorRT INT8 历史部署版本。

> 核心原则：保留可靠的原始米制深度，只在不可靠区域使用校准后的 LingBot 深度。

![方案概览](assets/slides/slide-01.png)

## 最终深度修复策略

1. 使用 `robbyant/lingbot-depth-postrain-dc-vitl14` 生成 LingBot depth。
2. 在每帧可靠像素上拟合 `raw_depth ≈ scale × lingbot_depth + bias`。
3. raw depth 默认保留，仅替换以下区域：
   - 无效深度及其 `3×3` 邻域；
   - 相对局部中值偏差超过 `max(6cm, 3.5% × raw)` 的噪声/异常点；
   - RGB 反光区域的 `11×11` 邻域；
   - raw 与 calibrated LingBot 差异超过 `max(8cm, 4% × raw)` 的区域。
4. 不使用时序滤波，也不使用 block-wise robust downsample。

最终质量配置：`max_side=320`、`tokens=300`、`max_depth=50m`、FP16。

详细说明见 [最终融合策略](docs/final_strategy.md)。

## 离线 Demo

三个 Demo 的融合有效深度均为 `100%`，像素替换比例约为 `16.6%–26.6%`。

| Demo | raw 有效率 | fusion 有效率 | 替换比例 | scale / bias |
|---|---:|---:|---:|---:|
| 115349 | 89.82% | 100% | 26.56% | 0.981 / +2.6cm |
| 183936 | 92.89% | 100% | 16.64% | 1.009 / -2.5cm |
| 194206 | 91.76% | 100% | 20.00% | 1.013 / -2.7cm |

### 115349

![115349 fusion comparison](assets/demos/115349.png)

### 183936

![183936 fusion comparison](assets/demos/183936.png)

### 194206

![194206 fusion comparison](assets/demos/194206.png)

对比视频：[GitHub Release: demo-v1](https://github.com/jingyu198/RGBD-repair/releases/tag/demo-v1)

## Orin 历史部署版本

机器人上跑通的是实时轻量版，并不是上面的最终 local fusion 质量版：

- TensorRT INT8，静态输入 `1×3×36×64`，`tokens=128`；
- PTQ Entropy calibration，使用三个 Demo 共 90 帧标定；
- 直接发布完整 LingBot depth，未做每帧 scale/bias 标定；
- LingBot depth 经原有处理链生成 MC 的 `32×18` depth input。

实测结果：最终 MC topic `29.967Hz`，TRT 推理 `23.134ms/frame`，节点总耗时 `33.137ms/frame`，relay `dropped=0`。

![Orin gray pipeline](assets/demos/orin_gray_pipeline.jpg)

详细说明见 [Orin 部署记录](docs/orin_deployment.md)。

## 运行最终融合 Demo

```bash
python src/lingbot_confidence_fusion_video.py \
  --repo /path/to/lingbot-depth \
  --input /path/to/synced.npz \
  --output-dir output/final_fusion \
  --stem 115349 \
  --max-side 320 --num-tokens 300 --max-depth 50 \
  --calibrate-lingbot --aggressive \
  --diff-abs 0.08 --diff-rel 0.04 \
  --noise-abs 0.06 --noise-rel 0.035 \
  --small-area 2000 --invalid-dilate 3 --specular-dilate 11 \
  --max-samples 6000 --calib-downscale 4
```

## 仓库内容

```text
assets/       README 图片
benchmarks/   离线与 Orin 实测 JSON
configs/      最终融合与 Orin 配置
deployment/   TensorRT 量化和 ROS2 部署代码
docs/         策略说明和 20260630 汇报 PPT
src/          标定与融合主脚本
tools/        原始处理链和 policy input 可视化
```

模型权重、ONNX、TensorRT engine、原始 RGB-D 数据和完整视频不进入 Git 历史。视频作为 GitHub Release 资产保存。

## 上游项目

- [robbyant/lingbot-depth](https://github.com/robbyant/lingbot-depth)
- [LingBot-Depth DC ViT-L/14 model](https://huggingface.co/robbyant/lingbot-depth-postrain-dc-vitl14)
