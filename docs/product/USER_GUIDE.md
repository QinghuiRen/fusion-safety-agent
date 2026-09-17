# Fusion Safety Agent 使用指南

## 1. 输入与输出

标准 Leaderboard 配置接收 6 路 RGB、1 路 LiDAR、4 路 Radar、GNSS、IMU、车速和
全局路线。扩展本地配置可使用 5 路 Radar，但官方 Leaderboard 传感器规则最多允许
4 路 Radar。

每个控制周期输出：

- 本车油门、制动、转向、倒挡和手刹命令；
- 目标 ID、类别、位置、速度、尺寸、协方差和证据可靠度；
- 0.5–3.0 秒未来轨迹及不确定性；
- TTC、安全裕度、最小预计间距、触发目标和风险等级；
- 推荐控制、实际控制、控制抑制原因和介入事件；
- 感知、传感器健康、时延和前端运行状态。

## 2. 运行模式

- `shadow`：计算并记录安全建议，不修改上游控制。默认且适合调试。
- `active`：允许纵向监督器限制油门或施加制动；不接管转向。

不要在未完成场景回归时把新配置直接切换为 Active。

## 3. 常用命令

```powershell
# 深度检查：模型证书、CARLA 路径和 WSL 响应
& .\.venv\Scripts\python.exe -m fusion_tracking_agent doctor

# 查看解析后的绝对配置
& .\.venv\Scripts\python.exe -m fusion_tracking_agent show-config

# 手动驾驶；duration=0 表示运行到 Esc
& .\.venv\Scripts\python.exe .\fusion_safety_agent.py manual `
  --duration 0 --npc 8 --manual-supervisor shadow
```

跨机器部署时建议显式配置外部运行时，源码和产品配置不会保存个人目录：

```powershell
$env:FUSION_SAFETY_CARLA_ROOT = "D:\runtime\CARLA_0.9.15"
$env:FUSION_SAFETY_CARLA_PYTHON = "D:\runtime\carla_python\.venv\Scripts\python.exe"
$env:FUSION_SAFETY_WSL_RUNTIME = "~/fusion_tracking_runtime/bevfusion"
```

如果未设置，Windows 启动器会检查常见 CARLA 目录以及桌面下的隔离运行环境；WSL
用户名始终在启动时通过 `$HOME` 动态解析。

## 4. 主要调参边界

编辑 `configs/product/fusion_safety_agent.json`：

- `detector_stride_frames`：BEVFusion 请求间隔；减小可提高源更新率，但增加 GPU 占用。
- `supervisor_max_fusion_age_s`：过期感知的控制准入上限。
- `gpu_arbiter_slow_threshold_s`：BEVFusion 长尾帧判定阈值。
- `gpu_arbiter_protection_results`：长尾发生后主感知独占保护次数。
- `vru_detector_stride_frames`：前视 VRU 辅助通道更新间隔。
- `supervisor_mode`：主纵向监督权限；交付默认 `shadow`。
- `lidar_safety_cage_mode`、`vru_safety_monitor_mode`：补充安全通道权限。

模型阈值、NMS、跟踪建轨门限和 Radar 关联半径由 Active 证书约束。修改这些字段后，
必须重新完成 Shadow 准入评测并签发证书，不能只改配置绕过质量门。

## 5. 运行结果

正式 Agent 默认输出至 `out/product/`。每次运行创建独立目录，核心文件包括逐帧 JSONL
和汇总 JSON。正式评测证据统一写入 `out/evaluation/product/`；旧的
`out/stage6_20/` 仅作为历史实验记录保留，读取工具仍兼容其旧文件名。

每次正式运行的核心产物为：

- `fusion_safety_frames.jsonl`：逐控制周期的输入状态、轨迹、风险、健康状态和控制结果；
- `fusion_safety_summary.json`：本次运行的配置、前端状态、时延、干预和安全指标汇总；
- `fusion_safety_replay.mp4`：启用回放渲染时生成的可视化视频；
- `fusion_safety_replay_summary.json`：回放帧数、编码参数和源数据说明。

正式输出使用带微秒的独立运行目录，避免连续启动时出现文件重名。读取与回放工具仍可
识别历史 Stage6.20 文件名，但正式包不会再产生旧命名。

## 6. 发布目录使用

源码仓库可复制，但仍需目标机器具备 CARLA 0.9.15、对应 Python 环境、WSL2、
CUDA 及已安装的 MMDetection3D/BEVFusion 运行环境。模型文件请按根目录 README
从私有 Release 下载，并核对 SHA-256；仓库不重复打包大型外部运行时。

```powershell
# 在仓库根目录执行环境、模型和证书检查
& .\.venv\Scripts\python.exe .\fusion_safety_agent.py doctor `
  --config .\configs\product\fusion_safety_agent.json
```
