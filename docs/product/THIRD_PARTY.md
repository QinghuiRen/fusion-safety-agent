# 第三方组件与实现归属

本仓库保留项目自有的适配、跟踪、预测、风险门控、闭环编排和评测代码；
不复制 CARLA、MMDetection3D、ScenarioRunner 或 carla_garage 的完整仓库。
第三方组件需要在仓库之外分别部署，并遵守各自许可证与模型使用条款。

| 组件 | 本项目中的作用 | 交付方式 |
| --- | --- | --- |
| [CARLA](https://github.com/carla-simulator/carla) 0.9.15 | 车辆动力学、传感器和仿真世界 | 用户单独安装 |
| [MMDetection3D BEVFusion](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion) | 相机与 LiDAR 的神经网络 BEV 检测 | WSL 独立环境；不随本仓库复制 |
| [carla_garage](https://github.com/autonomousvision/carla_garage) | Leaderboard Agent 接口与评测兼容资源 | 外部依赖；按用户环境配置 |
| [ScenarioRunner](https://github.com/carla-simulator/scenario_runner) | 标准场景回归接口 | 外部依赖；不属于产品运行代码 |
| [Autoware](https://github.com/autowarefoundation/autoware.universe) | AEB Shadow 对照的工程参考 | 本仓库为独立实现，未直接集成 Autoware 源码 |
| [nuScenes](https://www.nuscenes.org/) | 上游 BEVFusion 权重的预训练数据背景 | 数据不随仓库发布 |

仓库内 `fusion_tracking_bevfusion/` 是上游模型的输入输出桥接和本项目适配代码，
不是 BEVFusion 神经网络的原创声明。`fusion_tracking_carla/autoware_aeb_shadow.py`
是用于仿真对照的独立逻辑，不等于已验证的 Autoware AEB 实现。

当前私有仓库暂未授予公开复用许可。若计划公开，须先进行源码来源与项目成果归属、
预训练及微调权重再分发权限、第三方许可证兼容性审查，再确定本仓库许可证。
