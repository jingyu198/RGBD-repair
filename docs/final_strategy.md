# 最终 calibrated local fusion 策略

## 目标

RGB-D 传感器在反光地面、物体边缘和遮挡附近容易产生空洞、噪声和物理错误深度。最终方案不让网络覆盖全部传感器数据，而是将 LingBot-Depth 作为局部修复先验。

## 每帧处理

1. 输入 RGB、raw depth 和相机内参，运行 LingBot-Depth。
2. 从 raw 与 LingBot 都有效且相对可靠的像素中采样，最多 `6000` 点；标定阶段按 `4×` 降采样。
3. 鲁棒拟合：`raw_depth ≈ scale × lingbot_depth + bias`。
4. 将校准后的 LingBot depth 仅写入 replacement mask，其他位置保留 raw depth。

## replacement mask

| 区域 | 判定 |
|---|---|
| 无效区域 | raw depth 无效，并使用 `3×3` 膨胀覆盖邻域 |
| 噪声/异常点 | 相对 `5×5` 中值残差大于 `max(0.06m, 0.035 × raw)` |
| 反光区域 | RGB specular cue 经 `11×11` 膨胀，并与差异、异常或无效区域相交 |
| 明显不一致区域 | `abs(raw - calibrated_lingbot) > max(0.08m, 0.04 × raw)` |

`aggressive=true` 时，明显不一致区域可直接使用 calibrated LingBot 替换。

## 最终参数

```yaml
model: robbyant/lingbot-depth-postrain-dc-vitl14
max_side: 320
num_tokens: 300
max_depth_m: 50.0
precision: fp16
diff_abs_m: 0.08
diff_rel: 0.04
noise_abs_m: 0.06
noise_rel: 0.035
small_area: 2000
invalid_dilate: 3
specular_dilate: 11
calib_downscale: 4
max_samples: 6000
temporal_filter: false
blockwise_downsample: false
```

## 离线结果

三个 Demo 的 fused valid depth 均为 `100%`；替换比例为 `16.64%–26.56%`。该数据用于验证质量，不代表 Orin 实时性能。
