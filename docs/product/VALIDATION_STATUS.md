# Fusion Safety Agent 验证状态

## 已验证版本

- 验证日期：2026-09-15
- 实现指纹：`157852346BA023BE756200820DEBF16D5F9EBA152277001D010CD004812ED38B`
- 正式回归：9/9 个必测场景通过
- 路线完成率：9/9 均为 100%
- 碰撞：9/9 均为 0
- 感知运行错误：9/9 均为 0
- 主前端：`certified_bevfusion`
- BEVFusion 后端：WSL2 + MMDetection3D `projects/BEVFusion`
- 数据传输：常驻后端 `wsl_stdin_packet`
- Legacy 感知回退：0 个控制帧

完整机器可读证据位于
`out/evaluation/product/regression_20260915_105553/fusion_safety_regression_report.json`；正式发布包内固定副本为
`evidence/validated_regression_report.json`。

## 场景覆盖

验证矩阵覆盖 Town01 与 Town10HD_Opt，包括：

- Shadow 标称路线；
- Active 无危险负样本，用于验证无误介入；
- Active 跟车风险；
- Active 前车与静态障碍组合风险；
- Active 动态弱势交通参与者横穿；
- 前相机周期丢失；
- Radar 周期丢失；
- LiDAR 周期丢失。

三类故障场景均通过传感器健康门控。LiDAR 丢失用例出现 16 个过期感知帧，全部包含在 40 个受保护的控制抑制帧内，因此过期证据没有直接驱动 Active 控制。

## 本机时延口径

以 Town01 Active 无危险负样本为稳定性能样例：

- CARLA 仿真控制周期：20.00 Hz；
- 本机墙钟控制吞吐：12.15 Hz；
- 实时因子：0.608；
- BEVFusion 源数据更新率：2.50 Hz；
- BEVFusion 墙钟交付率：1.50 Hz；
- BEVFusion 稳态推理：P50 0.436 s，P95 0.511 s；
- 感知提交到交付：P50 0.577 s，P95 0.723 s；
- 融合结果年龄：P50 0.55 s，P95 0.80 s；
- 不含输出 I/O 的算法步骤：P50 9.62 ms，P95 19.80 ms；
- 隐私脱敏后的结构化输出总开销：P50 1.66 ms，P95 2.88 ms。

因此，“20 Hz”表示仿真时钟中的跟踪、风险和控制更新周期，不表示 BEVFusion 每秒推理 20 次，也不表示当前笔记本能以 20 Hz 墙钟速度完成整套 CARLA 仿真。重型神经网络低频更新，Radar/KF、风险模块和纵向控制在控制周期内进行外推、校正与门控。

## 交付定位与限制

当前交付是可重复运行、可审计、可调参，并具有 Active 纵向闭环的工程原型。它已经超出单次演示脚本的范围，但不是量产自动驾驶控制器。尚未完成实车传感器接入、HIL、道路测试、ISO 26262/SOTIF 安全认证、硬件冗余和量产时延预算验证。手动驾驶入口已保留，但发布包的键盘交互仍需在目标机器上由操作者进行一次最终验收。
