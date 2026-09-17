# 交付清单（v1.0.1）

## 私有 GitHub 仓库

- `fusion_safety_agent.py`：稳定 CLI / Leaderboard Agent 入口。
- `fusion_tracking_agent/`：运行编排、健康检查、手动和自动驾驶入口。
- `fusion_tracking_bevfusion/`：Windows—WSL BEVFusion 在线桥接和模型输出适配。
- `fusion_tracking_carla/`：CARLA 传感器、感知、跟踪、预测、风险与纵向监督。
- `fusion_tracking_core/`：可脱离 CARLA 的融合算法核心。
- `configs/product/`：默认产品配置；`configs/evaluation/`：评测配置。
- `tools/` 和 `tests/`：回归、验收、安装诊断及核心单元测试。
- `docs/product/`、`README.md`、`pyproject.toml`：使用、限制、来源和安装说明。
- `artifacts/bevfusion/active_runtime_certificate.json`：指定权重与参数的历史准入证书。
- `evidence/validated_regression_report.json`：已执行回归的脱敏报告；不是新环境的自动验收结果。
- `media/`：在 GitHub README 中直接播放的短预览动图。

## 同版本私有 GitHub Release 附件

- `fusion_safety_bevfusion_epoch_6.pth`：约 461 MiB 的微调模型权重。
- SHA-256：`ED7E8308D9C7654363A33094A9EEECE406913AD52CD0AA95F5D74476392E650A`。
- 使用前放入仓库根目录的 `artifacts/bevfusion/` 并核验哈希。权重不写入 Git 历史。
- `fusion_safety_agent-1.0.1-py3-none-any.whl`：可选的预构建 Python 安装包。
- `nuscenes_multisensor_tracking.mp4` 与 `carla_active_safety_dashboard.mp4`：
  H.264 完整演示视频；README 中的预览动图可点击打开视频。

## 不在交付包内

历史 Stage 实验、`out/` 运行结果、CARLA 与 nuScenes 数据、Python 虚拟环境、
外部 `carla_garage`/MMDetection3D 仓库、CUDA 编译产物以及本机绝对路径均不发布。
CARLA 0.9.15、WSL2 BEVFusion 运行时和其他外部依赖需要按使用指南单独部署。

仓库本身提供可安装源码和回归工具，但不保证在未安装外部仿真环境和权重的新机器上
克隆即跑通 CARLA 闭环。历史证据只适用于对应代码、权重和运行条件；部署后需要
重新执行 `doctor`、单元测试、Shadow 和 Active 场景回归。公开发布或附带权重分发
之前，需另行完成知识产权与许可证审查。
