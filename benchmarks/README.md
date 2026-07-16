# Benchmarks

- `offline/*.json`: final calibrated aggressive fusion，`max_side=320`、`tokens=300`。
- `orin/tokens128_deploy_10s.json`: no-record 10 秒实时测试。
- `orin/tokens128_visual_5s.json`: 带可视化记录的 5 秒测试。
- `orin/tokens128_engine_build.json`: INT8 engine 构建和 trtexec 数据。

离线推理数据来自质量验证环境；Orin 数据用于部署性能汇报，两者不要混用。
