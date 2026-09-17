"""CARLA 在线融合、跟踪、评测与显示的集中参数区。

调参原则：一次只改一组参数，并保留每次运行目录中的 JSONL/summary 作为对照。
长度单位均为 m，速度为 m/s，时间为 s，角度为 degree（字段名会标明）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from fusion_tracking_core import AssociationConfig, LidarCenterConfig, MarginSafetyConfig
from fusion_tracking_core.multicamera_fusion import MultiCameraFusionConfig


CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


@dataclass(frozen=True)
class RuntimeConfig:
    """仿真与运行参数。"""

    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 30.0
    map_load_timeout_s: float = 120.0  # 仅显式 --town 切图时使用。
    duration_s: float = 30.0  # 0 表示不限时，按 Esc 结束。
    fixed_delta_s: float = 0.05  # 20 Hz；降低可更流畅但增加计算压力。
    # 地图切换后、创建ego和传感器前先推进若干同步帧，让UE渲染资源完成初始化。
    # 普通阶段保持0；Stage6.14在官方Town05冷启动时显式开启。
    world_startup_warmup_ticks: int = 0
    seed: int = 20260822
    # 空字符串=沿用 CARLA 当前地图，启动最快；指定地图才触发耗时的 load_world。
    town: str = ""
    weather: str = "ClearNoon"
    spawn_npc_vehicles: int = 24
    # 正常评测为True；Stage6.16离线训练采集可设False，防止Traffic Manager
    # 在部分地图把刚生成的近场车辆重定位到远端。
    npc_autopilot_enabled: bool = True
    npc_spawn_radius_m: float = 140.0  # 优先在ego附近生成，保证短演示能看到目标。
    npc_min_spawn_distance_m: float = 12.0
    traffic_manager_port: int = 8000
    output_root: Path = Path("out/stage6_1")
    record_video: bool = True
    # 仅控制本地1600x900仪表盘的合成与显示；不会关闭CARLA六相机、LiDAR或Radar。
    # 自动批量评测可设False以移除可视化开销，手动驾驶保持True。
    render_dashboard: bool = True
    video_fps: float = 20.0
    # Windows全局键盘轮询不依赖Pygame焦点；解决CARLA窗口抢焦点后WASD失效。
    global_keyboard_input: bool = True
    enable_perception: bool = True  # False仅用于快速验证手动控制链。
    # 下面三项让同一套在线链路同时服务 Stage6.1 与 Stage6.2；6.1 默认值保持不变。
    stage_id: str = "stage6_1"
    run_name: str = ""  # 空=自动生成时间戳目录；批量评测时由 Stage6.2 指定。
    scenario_name: str = "interactive"


@dataclass(frozen=True)
class DrivingConfig:
    """ego 驾驶方式；感知始终是 shadow，不会擅自叠加制动。

    manual: 人用 WASD；autopilot: CARLA Traffic Manager 巡航；
    route_agent: CARLA BasicAgent 沿规划路线到指定终点。
    """

    mode: str = "manual"
    wait_for_enter: bool = True
    target_speed_kmh: float = 30.0
    traffic_manager_speed_difference_pct: float = 0.0
    traffic_manager_follow_distance_m: float = 2.5
    # BasicAgent 的障碍检测距离公式=base + speed_ratio*当前m/s；不要与TM跟车距离混用。
    route_vehicle_base_threshold_m: float = 8.0
    route_detection_speed_ratio_s: float = 1.2
    route_max_brake: float = 0.8
    # 只平滑BasicAgent巡航PID的基础油门/制动；安全监督器在其后独立仲裁，
    # 因此Active紧急制动不受这些速率限制。默认关闭，Stage6.18显式启用。
    smooth_route_longitudinal_control: bool = False
    route_throttle_rise_per_s: float = 1.5
    route_throttle_fall_per_s: float = 3.0
    route_brake_rise_per_s: float = 2.0
    route_brake_fall_per_s: float = 3.0
    # Optional jerk-limited cruise controller. BasicAgent keeps route/steering;
    # Active safety arbitration remains downstream and is never jerk-limited.
    route_speed_governor_enabled: bool = False
    route_speed_governor_kp: float = 0.65
    route_speed_governor_max_accel_mps2: float = 1.6
    route_speed_governor_max_decel_mps2: float = 2.0
    route_speed_governor_jerk_mps3: float = 1.5
    route_speed_governor_throttle_feedforward: float = 0.38
    route_speed_governor_throttle_per_accel: float = 0.18
    route_speed_governor_max_throttle: float = 0.65
    route_speed_governor_brake_per_decel: float = 0.05
    route_speed_governor_max_brake: float = 0.12
    # Stage6.8确定性安全场景使用：只保留BasicAgent的路线/横向跟踪和巡航PID，
    # 禁用其基于CARLA actor真值的车辆避障。此时前车制动只能由本项目感知、预测
    # 与纵向安全监督器给出，避免把仿真真值能力误算成我们的算法效果。
    route_ignore_vehicles: bool = False
    # 确定性短场景可同时忽略随机交通灯相位，保证Baseline/Shadow/Active的输入条件一致。
    # 常规道路闭环应保持False，让BasicAgent遵守交通灯。
    route_ignore_traffic_lights: bool = False
    route_target_distance_m: float = 250.0
    route_min_distance_m: float = 100.0
    route_goal_tolerance_m: float = 5.0
    stop_on_route_completion: bool = True
    # BasicAgent 在 CARLA 安装目录的 PythonAPI/carla 下，独立 venv 默认没有加入 sys.path。
    carla_agents_root: Path | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Stage6.2 在线验收指标阈值；只用于判读，不改变车辆控制。"""

    lane_departure_offset_m: float = 1.75
    critical_risk_threshold: float = 75.0
    max_p95_jerk_mps3: float = 8.0
    # CARLA 20Hz原始加速度含物理步进抖动，且二维向量差分会把转弯向心加速度误计为
    # 纵向舒适性。乘员舒适度使用车身纵向加速度的一阶低通结果；原始二维jerk仍保留诊断。
    comfort_acceleration_filter_tau_s: float = 0.30
    max_collision_count: int = 0
    max_sensor_timeout_frames: int = 2
    min_route_completion_pct: float = 95.0
    # CARLA一次持续接触会连续触发多帧collision callback；同一actor无间断接触只算一次事故。
    collision_dedup_window_s: float = 1.0
    # CARLA 真值仅用于离线/在线评分，严禁回流到感知、风险或控制。
    enable_oracle_perception_metrics: bool = False
    oracle_max_range_m: float = 65.0
    oracle_match_gate_m: float = 6.0
    oracle_exclude_tentative_tracks: bool = True
    oracle_id_switch_gap_s: float = 1.5
    oracle_frame_tolerance_s: float = 0.05
    oracle_prediction_horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class SensorConfig:
    """传感器频率和采样密度；六相机用于展示及检测。"""

    # 400x240 与仪表盘单相机格尺寸一致；MobileNet内部仍缩放到约320像素。
    camera_width: int = 400
    camera_height: int = 240
    camera_fov_deg: float = 90.0
    sensor_tick_s: float = 0.05
    # None=all sensors follow sensor_tick_s.  A larger value enables a genuine
    # multi-rate graph: RGB is rendered slowly while LiDAR/Radar and control
    # keep the base rate.  Keep it an integer multiple of sensor_tick_s so the
    # six cameras remain time-aligned.
    camera_sensor_tick_s: float | None = None
    lidar_channels: int = 32
    lidar_range_m: float = 80.0
    lidar_points_per_second: int = 160_000
    lidar_rotation_hz: float = 20.0
    radar_range_m: float = 70.0
    radar_horizontal_fov_deg: float = 80.0
    radar_vertical_fov_deg: float = 20.0
    radar_points_per_second: int = 1000
    # CARLA 首帧会编译/加载渲染资源，等待明显长于稳定运行阶段。
    startup_sensor_wait_s: float = 15.0
    max_sensor_wait_s: float = 5.0
    sensor_warmup_retries: int = 2
    # 0 表示沿用旧行为：六相机在同一时刻创建。正整数表示把六相机分批创建，
    # 每批创建后先推进一帧并确认该批已返回图像，再创建下一批。此选项用于规避
    # Windows CARLA 在首次载入高负载地图时集中初始化多路 RGB 渲染资源的崩溃；
    # 它不减少相机数量，也不改变正式运行时的 20 Hz 同步输入。
    camera_startup_batch_size: int = 0
    camera_startup_batch_timeout_s: float = 15.0
    # 空元组沿用标准环视顺序。调试 CARLA/UE4 渲染崩溃时可传入六相机的完整排列，
    # 只改变冷启动先后，不改变通道名、外参、正式输入数量或后续算法接口。
    camera_startup_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerceptionConfig:
    """检测与点云抬升参数。

    detector_stride_frames 控制检测频率：20 Hz 仿真下设 10 即约 2 Hz。
    若实时性不足先增大 stride，再降低图像尺寸；不要先牺牲检测阈值。
    """

    detector_model: str = "fasterrcnn_resnet50_fpn_v2"
    detector_score_threshold: float = 0.55
    detector_stride_frames: int = 10
    # 六路图像始终由SensorRig采集并保留给仪表盘/接口；这里只决定哪些相机进入
    # 重型2D检测器。纵向安全档可用前向三相机降低GPU占用，全景感知仍用默认六路。
    detector_camera_channels: tuple[str, ...] = CAMERA_CHANNELS
    # fp32=稳定基线；amp_fp16=仅CUDA前向使用自动混合精度。若设备或算子不兼容，
    # CameraRoadUserDetector会记录原因并自动回退FP32，不能静默伪报FP16收益。
    detector_precision: str = "fp32"
    detector_batch_size: int = 2  # 六图分批，降低 Faster R-CNN 峰值显存。
    # 在开始计时前真实跑一次前向，触发 CUDA kernel/显存缓存；不会产生算法输入。
    detector_warmup_inference: bool = True
    detector_startup_timeout_s: float = 180.0  # 首次权重载入/下载只发生一次。
    detector_max_objects_per_camera: int = 30
    lidar_projection_margin_px: int = 2
    # 2D框内LiDAR很稀疏或前后景深度跨度过大时，点云不足以可靠估计目标尺寸。
    # 此时仅对尺寸使用类别族先验；中心和位置协方差仍来自真实LiDAR/Radar证据。
    # 调参顺序：先看lifting_evidence.depth_spread_m，再调整跨度门限；不要为单帧放宽。
    extent_prior_min_lidar_points: int = 6
    extent_incoherent_depth_spread_m: float = 4.0
    extent_prior_max_scale: float = 1.8
    # 远距2D框没有足够LiDAR点时，可用同框Radar空间回波形成高不确定度3D观测。
    # 默认关闭，先在Stage6.14官方场景做受控A/B；绝不允许纯视觉单目深度建轨。
    radar_only_lift_enabled: bool = False
    radar_only_lift_min_visual_score: float = 0.65
    radar_only_lift_min_points: int = 2
    # 远距VRU框面积小且类别更易与车辆混淆，要求更多Radar回波才允许建观测。
    radar_only_lift_vru_min_points: int = 3
    radar_only_lift_min_range_m: float = 25.0
    radar_only_lift_cluster_radius_m: float = 2.5
    radar_only_lift_position_std_m: float = 2.5
    # Optional calibrated monocular fallback for vulnerable road users only.
    # It intersects the detected box footpoint ray with the local road plane;
    # no actor/map/scenario truth or learned monocular depth is used.  Keep it
    # disabled for the general detector and enable it only in a dedicated,
    # independently certified VRU safety channel.
    ground_plane_vru_lift_enabled: bool = False
    # In a dedicated VRU channel, prefer the calibrated footpoint over LiDAR
    # points projected into the 2D box.  This prevents an occluding prop/wall
    # behind a pedestrian from becoming the pedestrian's apparent 3D centre.
    # The general road-user detector keeps this disabled.
    ground_plane_vru_prefer_over_lidar: bool = False
    ground_plane_vru_min_visual_score: float = 0.70
    ground_plane_vru_min_range_m: float = 2.0
    ground_plane_vru_max_range_m: float = 30.0
    ground_plane_vru_max_lateral_m: float = 5.0
    ground_plane_vru_horizon_margin_px: float = 8.0
    ground_plane_vru_position_std_base_m: float = 1.2
    ground_plane_vru_position_std_per_range: float = 0.08
    # CAM_BACK 会看到自车尾部。多边形使用归一化图像坐标 (x, y)，覆盖固定的
    # 自车可见轮廓；更换 ego 车型或后相机安装位姿后，需要用仪表盘截图重标。
    ego_body_mask_channels: tuple[str, ...] = ("CAM_BACK",)
    ego_body_mask_polygon_norm: tuple[tuple[float, float], ...] = (
        (0.36, 0.50), (0.64, 0.50), (0.64, 0.75),
        (0.82, 1.00), (0.18, 1.00), (0.36, 0.75),
    )
    # 检测器输入端先对该区域做修复式遮罩；仪表盘仍使用未修改的原始相机图像。
    ego_body_detector_mask_dilation_px: int = 4
    ego_body_detector_inpaint_radius_px: int = 5
    # 后置框过滤是第二道保险，处理检测器仍对遮罩边缘给出车辆框的情况。
    ego_body_box_min_overlap_ratio: float = 0.55
    ego_body_box_min_center_y_ratio: float = 0.62
    lidar_center: LidarCenterConfig = field(default_factory=lambda: LidarCenterConfig(
        mode="robust", min_points=3, ground_min_z_m=-1.25,
        depth_cluster_gap_m=1.2, robust_center_blend=0.55,
    ))
    multicamera: MultiCameraFusionConfig = field(
        default_factory=lambda: MultiCameraFusionConfig(mode="overlap_aware")
    )


@dataclass(frozen=True)
class TrackerConfig:
    """流式 CV-KF、关联和轨迹生命周期参数。"""

    process_accel_std_mps2: float = 3.0  # 大则更跟手，小则更平滑。
    initial_position_std_m: float = 2.0
    initial_velocity_std_mps: float = 8.0
    # 低分检测仍可续接已有轨迹，但只有达到该分数才允许创建新ID。
    # 默认0保持Stage6.1/6.2/6.3旧行为；Stage6.5可显式设为0.25。
    new_track_score_threshold: float = 0.0
    position_observation_std_m: float = 1.2
    radial_velocity_std_mps: float = 1.8
    radar_association_gate_m: float = 3.2
    radar_min_support: int = 1
    radial_outlier_gate_sigma: float = 3.5
    # 多速率模式：视觉融合批次之间，用当前帧 Radar 对只读预测快照做速度修正。
    # 它不改变基础轨迹时间线，避免异步视觉结果形成乱序更新。
    enable_live_radar_refinement: bool = False
    live_radar_association_gate_m: float = 3.2
    # 只读视觉KF之外保留短时Radar快状态，避免每个20Hz帧都弹回旧视觉速度。
    # 新视觉命中会立即重置；超过该时长没有可用Radar测量也会失效。
    live_radar_memory_max_age_s: float = 1.0
    # 二阶段恢复只处理第一阶段未匹配项，用于吸收 LiDAR 中心偶发跳动。
    # Stage6.1/6.2 默认关闭，Stage6.3 开启，便于保留旧基线。
    enable_recovery_association: bool = False
    recovery_vehicle_gate_m: float = 9.0
    recovery_vru_gate_m: float = 4.0
    recovery_max_size_log_ratio: float = 1.2
    recovery_radial_innovation_gate_mps: float = 6.0
    # 仅为显式Radar-only远距观测延长连续性；普通LiDAR轨迹仍使用下方原门限。
    enable_radar_only_track_continuity: bool = False
    radar_only_max_missed_updates: int = 12
    # Radar-only轨迹可在内部保留更久以恢复原ID，但过期预测不能持续送入安全链。
    # 20Hz主循环中3次约0.15s；实际missed只在新融合批次到达时增长，通常约0.75s。
    radar_only_publish_max_missed_updates: int = 3
    radar_only_recovery_vehicle_gate_m: float = 18.0
    radar_only_recovery_radial_innovation_gate_mps: float = 12.0
    min_confirmed_hits: int = 2
    max_missed_updates: int = 4
    max_prediction_age_s: float = 2.5
    association: AssociationConfig = field(default_factory=lambda: AssociationConfig(
        mode="mahalanobis",
        vehicle_hard_gate_m=7.0,
        vru_hard_gate_m=3.5,
        position_mahalanobis_gate=4.0,
        size_log_ratio_weight=0.25,
    ))


@dataclass(frozen=True)
class DashboardConfig:
    """固定分区布局，避免相机、BEV 和状态面板互相覆盖。"""

    width_px: int = 1600
    height_px: int = 900
    left_width_px: int = 800
    center_width_px: int = 480
    top_height_px: int = 720
    bev_forward_m: float = 65.0
    bev_backward_m: float = 15.0
    bev_half_width_m: float = 32.0
    path_history_length: int = 1200
    telemetry_history_s: float = 15.0
    # BEV实线表示目标自身的运动轨迹。低于该速度视为静态或滤波微抖动：
    # 目标框与风险仍显示，但不画未来轨迹线。这里仅影响显示，不影响安全预测。
    trajectory_display_min_speed_mps: float = 0.75
    trajectory_display_horizon_s: float = 3.0
    trajectory_display_step_s: float = 0.5


@dataclass(frozen=True)
class Stage61Config:
    """Stage6.1 完整配置；用于模块调用、JSON 记录和复现实验。"""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    driving: DrivingConfig = field(default_factory=DrivingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    safety: MarginSafetyConfig = field(default_factory=MarginSafetyConfig)

    def validate(self) -> None:
        if self.runtime.fixed_delta_s <= 0:
            raise ValueError("fixed_delta_s 必须大于 0")
        if self.sensors.sensor_tick_s <= 0:
            raise ValueError("sensor_tick_s必须大于0")
        camera_tick = self.sensors.camera_sensor_tick_s
        if camera_tick is not None:
            if camera_tick < self.sensors.sensor_tick_s:
                raise ValueError("camera_sensor_tick_s不能小于sensor_tick_s")
            ratio = camera_tick / self.sensors.sensor_tick_s
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError("camera_sensor_tick_s必须是sensor_tick_s的整数倍")
        if self.runtime.world_startup_warmup_ticks < 0:
            raise ValueError("world_startup_warmup_ticks 不能为负")
        if self.runtime.record_video and not self.runtime.render_dashboard:
            raise ValueError("record_video=True需要render_dashboard=True")
        if self.driving.wait_for_enter and not self.runtime.render_dashboard:
            raise ValueError("wait_for_enter=True需要可见仪表盘")
        if self.perception.detector_stride_frames < 1:
            raise ValueError("detector_stride_frames 必须 >= 1")
        if self.sensors.camera_startup_batch_size < 0:
            raise ValueError("camera_startup_batch_size 不能为负")
        if self.sensors.camera_startup_batch_timeout_s <= 0:
            raise ValueError("camera_startup_batch_timeout_s 必须大于 0")
        startup_order = self.sensors.camera_startup_order
        if startup_order:
            if (
                len(startup_order) != len(CAMERA_CHANNELS)
                or len(set(startup_order)) != len(CAMERA_CHANNELS)
                or set(startup_order) != set(CAMERA_CHANNELS)
            ):
                raise ValueError("camera_startup_order必须是六相机通道的完整无重复排列")
        detector_channels = self.perception.detector_camera_channels
        if not detector_channels:
            raise ValueError("detector_camera_channels不能为空")
        if len(set(detector_channels)) != len(detector_channels):
            raise ValueError("detector_camera_channels不能重复")
        unknown_channels = set(detector_channels) - set(CAMERA_CHANNELS)
        if unknown_channels:
            raise ValueError(
                f"detector_camera_channels包含未知相机: {sorted(unknown_channels)}"
            )
        if self.perception.extent_prior_min_lidar_points < 1:
            raise ValueError("extent_prior_min_lidar_points必须 >= 1")
        if self.perception.extent_incoherent_depth_spread_m <= 0:
            raise ValueError("extent_incoherent_depth_spread_m必须大于0")
        if self.perception.extent_prior_max_scale < 1:
            raise ValueError("extent_prior_max_scale必须 >= 1")
        if not 0.0 <= self.perception.radar_only_lift_min_visual_score <= 1.0:
            raise ValueError("radar_only_lift_min_visual_score必须在[0,1]")
        if self.perception.radar_only_lift_min_points < 1:
            raise ValueError("radar_only_lift_min_points必须 >= 1")
        if self.perception.radar_only_lift_vru_min_points < 1:
            raise ValueError("radar_only_lift_vru_min_points必须 >= 1")
        if self.perception.radar_only_lift_min_range_m <= 0:
            raise ValueError("radar_only_lift_min_range_m必须大于0")
        if self.perception.radar_only_lift_cluster_radius_m <= 0:
            raise ValueError("radar_only_lift_cluster_radius_m必须大于0")
        if self.perception.radar_only_lift_position_std_m <= 0:
            raise ValueError("radar_only_lift_position_std_m必须大于0")
        if not 0.0 < self.perception.ground_plane_vru_min_visual_score <= 1.0:
            raise ValueError("ground_plane_vru_min_visual_score must be in (0,1]")
        if not (
            0.0 < self.perception.ground_plane_vru_min_range_m
            < self.perception.ground_plane_vru_max_range_m
        ):
            raise ValueError("ground-plane VRU range must satisfy 0 < min < max")
        ground_plane_positive = (
            self.perception.ground_plane_vru_max_lateral_m,
            self.perception.ground_plane_vru_horizon_margin_px,
            self.perception.ground_plane_vru_position_std_base_m,
            self.perception.ground_plane_vru_position_std_per_range,
        )
        if any(value <= 0 for value in ground_plane_positive):
            raise ValueError("ground-plane VRU gates and uncertainty must be >0")
        if self.perception.detector_precision not in {"fp32", "amp_fp16"}:
            raise ValueError("detector_precision必须是fp32或amp_fp16")
        if self.dashboard.left_width_px + self.dashboard.center_width_px >= self.dashboard.width_px:
            raise ValueError("dashboard 左区+中区必须给右侧状态区留出空间")
        if self.dashboard.trajectory_display_min_speed_mps < 0:
            raise ValueError("trajectory_display_min_speed_mps不能为负")
        if (
            self.dashboard.trajectory_display_horizon_s <= 0
            or self.dashboard.trajectory_display_step_s <= 0
        ):
            raise ValueError("轨迹显示时域和步长必须为正")
        if self.driving.mode not in {"manual", "autopilot", "route_agent"}:
            raise ValueError("driving.mode 必须是 manual/autopilot/route_agent")
        if self.driving.target_speed_kmh <= 0:
            raise ValueError("target_speed_kmh 必须大于 0")
        smoothing_rates = (
            self.driving.route_throttle_rise_per_s,
            self.driving.route_throttle_fall_per_s,
            self.driving.route_brake_rise_per_s,
            self.driving.route_brake_fall_per_s,
        )
        if any(value <= 0 for value in smoothing_rates):
            raise ValueError("BasicAgent纵向控制平滑速率必须>0")
        governor_positive = (
            self.driving.route_speed_governor_kp,
            self.driving.route_speed_governor_max_accel_mps2,
            self.driving.route_speed_governor_max_decel_mps2,
            self.driving.route_speed_governor_jerk_mps3,
            self.driving.route_speed_governor_throttle_per_accel,
            self.driving.route_speed_governor_max_throttle,
            self.driving.route_speed_governor_brake_per_decel,
            self.driving.route_speed_governor_max_brake,
        )
        if any(value <= 0 for value in governor_positive):
            raise ValueError("route speed governor gains and limits must be > 0")
        if not 0 <= self.driving.route_speed_governor_throttle_feedforward <= 1:
            raise ValueError("route speed governor throttle feedforward must be in [0, 1]")
        if self.tracker.live_radar_association_gate_m <= 0:
            raise ValueError("live_radar_association_gate_m 必须大于 0")
        if self.tracker.live_radar_memory_max_age_s <= 0:
            raise ValueError("live_radar_memory_max_age_s 必须大于 0")
        if self.tracker.recovery_vehicle_gate_m <= 0 or self.tracker.recovery_vru_gate_m <= 0:
            raise ValueError("recovery association 距离门限必须大于 0")
        if self.tracker.recovery_max_size_log_ratio <= 0:
            raise ValueError("recovery_max_size_log_ratio 必须大于 0")
        if self.tracker.recovery_radial_innovation_gate_mps <= 0:
            raise ValueError("recovery_radial_innovation_gate_mps 必须大于 0")
        if self.tracker.radar_only_max_missed_updates < self.tracker.max_missed_updates:
            raise ValueError("radar_only_max_missed_updates不能小于普通轨迹寿命")
        if not 0 <= self.tracker.radar_only_publish_max_missed_updates <= self.tracker.radar_only_max_missed_updates:
            raise ValueError("radar_only_publish_max_missed_updates必须在[0, radar_only_max_missed_updates]")
        if self.tracker.radar_only_recovery_vehicle_gate_m <= 0:
            raise ValueError("radar_only_recovery_vehicle_gate_m必须大于0")
        if self.tracker.radar_only_recovery_radial_innovation_gate_mps <= 0:
            raise ValueError("radar_only恢复径向创新门限必须大于0")
        if self.evaluation.oracle_max_range_m <= 0:
            raise ValueError("oracle_max_range_m 必须大于 0")
        if self.evaluation.oracle_match_gate_m <= 0:
            raise ValueError("oracle_match_gate_m 必须大于 0")
        horizons = self.evaluation.oracle_prediction_horizons_s
        if any(value <= 0 for value in horizons) or len(set(horizons)) != len(horizons):
            raise ValueError("oracle_prediction_horizons_s 必须为互不重复的正数")
        self.safety.validate()

    def to_dict(self) -> dict:
        """返回可 JSON 序列化配置。"""

        result = asdict(self)
        result["runtime"]["output_root"] = str(self.runtime.output_root)
        agents_root = self.driving.carla_agents_root
        result["driving"]["carla_agents_root"] = None if agents_root is None else str(agents_root)
        return result


@dataclass(frozen=True)
class Stage62Config(Stage61Config):
    """Stage6.2 自动驾驶/路线驾驶与场景评测默认配置。"""

    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(
        stage_id="stage6_2",
        output_root=Path("out/stage6_2"),
        scenario_name="closed_loop",
    ))
    driving: DrivingConfig = field(default_factory=lambda: DrivingConfig(
        mode="route_agent",
        wait_for_enter=False,
    ))


@dataclass(frozen=True)
class Stage63Config(Stage62Config):
    """Stage6.3：真值隔离评分、分段延迟和多速率 Radar 修正。"""

    runtime: RuntimeConfig = field(default_factory=lambda: RuntimeConfig(
        stage_id="stage6_3",
        output_root=Path("out/stage6_3"),
        scenario_name="multirate_perception",
    ))
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig(
        enable_oracle_perception_metrics=True,
    ))
    perception: PerceptionConfig = field(default_factory=lambda: PerceptionConfig(
        detector_model="fasterrcnn_mobilenet_v3_large_320_fpn",
        detector_stride_frames=10,
    ))
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(
        enable_live_radar_refinement=True,
        enable_recovery_association=True,
    ))
