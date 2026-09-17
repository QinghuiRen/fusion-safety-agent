# 模型说明与使用边界

## 身份和来源

本产品的视觉—LiDAR 前端使用 MMDetection3D `projects/BEVFusion` 模型结构。
`fusion_safety_bevfusion_epoch_6.pth` 是基于上游 nuScenes 预训练权重、在 CARLA
合成数据上继续训练六个 epoch 的项目权重，不是从零训练，也不是本仓库原创的
BEV 特征融合网络。模型权重与本仓库源码分开交付，仅在私有 `v1.0.1` Release
中提供。公开分发前须先确认上游权重、训练数据和项目代码的使用与再分发权限。

- 文件：`artifacts/bevfusion/fusion_safety_bevfusion_epoch_6.pth`
- SHA-256：`ED7E8308D9C7654363A33094A9EEECE406913AD52CD0AA95F5D74476392E650A`
- 在线推理：WSL2 Ubuntu 22.04 中的 MMDetection3D/BEVFusion；CARLA 主进程运行在 Windows。
- 输入：同步六相机图像、LiDAR 点云和传感器标定；Radar 不进入 BEVFusion 网络，
  而是在检测结果之后参与速度关联、跟踪和风险评估。
- 输出：三维目标候选及置信度；本项目再生成轨迹、预测、安全证据和纵向控制建议。

## 已验证范围

`artifacts/bevfusion/active_runtime_certificate.json` 记录了一个固定运行契约下
的准入验证：701 个评测帧，位置 RMSE 约 2.71 m、速度 RMSE 约 2.25 m/s、
精确率约 0.82、召回率约 0.51；该记录中的平均感知时延约 0.765 s、
视觉更新率约 1.29 Hz。这是特定 CARLA 场景、参数和机器上的历史验证，
不能外推为跨域性能，也不表示 BEVFusion 本身以 20 Hz 实时推理。
更完整的系统级回归边界见 `VALIDATION_STATUS.md`；证书只对其指定的权重哈希
与运行参数有效。

## 风险和限制

- CARLA 合成影像、天气、车型和传感器噪声与真实道路存在域差异。
- 小目标、遮挡、远距和异常光照下可能漏检；后级 Radar 与 LiDAR 安全通道
  不能替代独立的量产级功能安全设计。
- 仅支持仿真研究和工程验证，禁止把该权重或输出直接用于真实车辆控制。
- 权重文件是可执行反序列化格式；仅从已核对来源下载并在加载前验证 SHA-256。

## 再验证要求

更换地图、天气、相机参数、模型权重、置信度门限或控制策略后，应重新运行
单元测试、Shadow 对照、正负安全场景及故障注入回归。只有在准入门槛全部重新
满足后才应在 CARLA 中启用 Active 模式。
