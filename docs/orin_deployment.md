# Orin 历史部署记录

## 版本说明

Orin 上部署的是为 30Hz 实时性优化的轻量版，与最终 calibrated local fusion 质量版不同：

- TensorRT INT8；
- `tokens=128`；
- 静态输入 `1×3×36×64`；
- 完整 LingBot depth 输出；
- 不做每帧 metric scale/bias 标定；
- 下游 MC depth input 为 `32×18`。

## INT8 量化

- TensorRT PTQ / Entropy calibration；
- 使用三个离线 Demo，共 90 帧 calibration samples；
- engine: `lingbot_tokens128_36x64_int8.engine`；
- trtexec throughput: `46.8571 FPS`；
- trtexec GPU compute: `21.2486 ms`。

## 机器人实测

| 指标 | 结果 |
|---|---:|
| 最终 MC topic | 29.967 Hz |
| LingBot 中间 topic | 29.969 Hz |
| 内部 publish FPS | 30.00 FPS |
| TRT 推理 | 23.134 ms/frame |
| 预处理 | 1.431 ms/frame |
| 节点总耗时 | 33.137 ms/frame |
| Relay | forwarded=721, dropped=0 |

原始测量文件位于 `benchmarks/orin/`。

## 高 token 量化结论

`tokens=300` 时，Orin 上 FP16 约为 `16.30 FPS`，INT8 PTQ 约为 `15.64 FPS`。部分 ViT 层回退到非 INT8 精度，INT8 没有带来速度收益，因此高质量版本更适合 FP16，实时 30Hz 使用 tokens128 INT8。
