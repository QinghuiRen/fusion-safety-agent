"""在隔离的 MMDetection3D 环境中运行官方 BEVFusion。

此脚本刻意不被当前项目环境 import。它应由 WSL/Linux 的 BEVFusion Python
解释器执行，把结果写成 ``fusion_tracking.bevfusion.v1`` JSON 后退出。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path


NUSCENES_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
SCHEMA_VERSION = "fusion_tracking.bevfusion.v1"
SEQUENCE_SCHEMA_VERSION = "fusion_tracking.bevfusion.sequence.v1"
OFFICIAL_CHECKPOINT_SHA256 = (
    "E68C6C084D8971F4F879751B0ED3981F575C24EE3CF385A1E39197C26A9BB3AB"
)


def _activate_repo(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    if not (root / "mmdet3d" / "__init__.py").is_file():
        raise FileNotFoundError(f"不是 MMDetection3D 仓库: {root}")
    os.chdir(root)
    sys.path.insert(0, str(root))
    return root


def _imports():
    import mmengine
    import numpy as np
    import torch
    from mmcv.transforms import Compose
    from mmengine.dataset import pseudo_collate
    from mmdet3d.structures import get_box_type

    # 触发项目注册；缺少自定义 CUDA 算子时会在这里给出明确错误。
    import projects.BEVFusion.bevfusion  # noqa: F401

    return mmengine, np, torch, Compose, pseudo_collate, get_box_type


def preflight(repo_root: str | Path) -> int:
    _activate_repo(repo_root)
    try:
        mmengine, _, torch, *_ = _imports()
        import mmcv
        import mmdet
        import mmdet3d
        from projects.BEVFusion.bevfusion.ops.bev_pool import bev_pool  # noqa: F401
        from projects.BEVFusion.bevfusion.ops.voxel import Voxelization  # noqa: F401
    except Exception as exc:
        print(f"[BEVFusion 后端预检失败] {type(exc).__name__}: {exc}")
        return 2
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    print(
        "[BEVFusion 后端预检通过] "
        f"torch={torch.__version__} mmcv={mmcv.__version__} "
        f"mmdet={mmdet.__version__} mmdet3d={mmdet3d.__version__} "
        f"mmengine={mmengine.__version__} device={device} capability={capability}"
    )
    return 0


def prepare_infos(repo_root: str | Path, data_root: str | Path, prefix: str) -> int:
    root = _activate_repo(repo_root)
    _imports()
    from tools.dataset_converters import nuscenes_converter
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

    data_root = Path(data_root).resolve()
    # 上游v2转换器内部固定从 ./data/nuscenes 读取标定；给它建立只读数据链接，
    # 不复制数据，也不改变nuScenes目录内容。
    data_link = root / "data" / "nuscenes"
    data_link.parent.mkdir(parents=True, exist_ok=True)
    if data_link.is_symlink():
        if data_link.resolve() != data_root:
            data_link.unlink()
            data_link.symlink_to(data_root, target_is_directory=True)
    elif data_link.exists():
        if data_link.resolve() != data_root:
            raise ValueError(f"{data_link} 已存在且不指向 {data_root}")
    else:
        data_link.symlink_to(data_root, target_is_directory=True)
    nuscenes_converter.create_nuscenes_infos(
        str(data_root), prefix, version="v1.0-mini", max_sweeps=10
    )
    for split in ("train", "val"):
        info_file = data_root / f"{prefix}_infos_{split}.pkl"
        update_pkl_infos("nuscenes", out_dir=str(data_root), pkl_path=str(info_file))
        print(f"[nuScenes info] {info_file}")
    print(f"[说明] 只生成推理 info，没有建立训练用 GT database | repo={root}")
    return 0


def _resolve_sensor_path(
    data_root: Path, value: str | Path, fallback_prefix: str | Path
) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    direct = data_root / path
    if direct.is_file():
        return str(direct.resolve())
    fallback = data_root / fallback_prefix / path.name
    return str(fallback.resolve())


def _absolute_data_paths(data_info: dict, data_root: Path) -> dict:
    """把 v2 info 中相对路径转绝对路径，包含关键帧和历史 sweep。"""

    item = deepcopy(data_info)
    lidar = item["lidar_points"]
    lidar["lidar_path"] = _resolve_sensor_path(
        data_root, lidar["lidar_path"], "samples/LIDAR_TOP"
    )
    for channel, image in item.get("images", {}).items():
        image["img_path"] = _resolve_sensor_path(
            data_root, image["img_path"], f"samples/{channel}"
        )
    for sweep in item.get("lidar_sweeps", []):
        sweep_lidar = sweep["lidar_points"]
        sweep_lidar["lidar_path"] = _resolve_sensor_path(
            data_root, sweep_lidar["lidar_path"], "sweeps/LIDAR_TOP"
        )
    return item


def _select_data_info(mmengine, info_files: list[Path], sample_token: str) -> dict:
    matches = []
    for info_file in info_files:
        payload = mmengine.load(str(info_file))
        matches.extend(
            item
            for item in payload["data_list"]
            if str(item.get("token", item.get("sample_idx", ""))) == sample_token
        )
    if len(matches) != 1:
        raise ValueError(
            f"sample_token={sample_token} 在 {info_files} 中命中 {len(matches)} 次"
        )
    return matches[0]


def _index_data_infos(mmengine, info_files: list[Path]) -> dict[str, dict]:
    """Load each info file once and index v2 records by nuScenes token."""

    indexed: dict[str, dict] = {}
    for info_file in info_files:
        payload = mmengine.load(str(info_file))
        for item in payload["data_list"]:
            token = str(item.get("token", item.get("sample_idx", "")))
            if not token:
                continue
            if token in indexed:
                raise ValueError(f"sample_token={token} 在info文件中重复")
            indexed[token] = item
    return indexed


def _disable_redundant_backbone_init(model_config, checkpoint_source, torch) -> str:
    """完整检测器权重已有Swin参数时，禁止模型构造阶段再次联网下载预训练权重。"""

    payload = (
        checkpoint_source
        if isinstance(checkpoint_source, dict)
        else torch.load(
            str(checkpoint_source), map_location="cpu", weights_only=False
        )
    )
    state_dict = payload.get("state_dict", payload)
    backbone_keys = [key for key in state_dict if key.startswith("img_backbone.")]
    if not backbone_keys:
        raise ValueError("checkpoint不含img_backbone权重，不能跳过Swin预训练初始化")
    init_cfg = model_config.model.img_backbone.get("init_cfg")
    source = ""
    if init_cfg:
        source = str(init_cfg.get("checkpoint", ""))
        model_config.model.img_backbone.init_cfg = None
    return source


def _convert_sparseconv_checkpoint_layout(state_dict, model_state_dict) -> list[str]:
    """Convert spconv2 kernels for the MMCV SparseConv fallback in memory.

    Official BEVFusion checkpoints may store 5-D sparse kernels as
    ``[out, D, H, W, in]`` (spconv2), while the MMCV fallback used on this
    machine expects ``[D, H, W, in, out]``.  Conversion is deliberately
    restricted to ``pts_middle_encoder`` and accepted only when the permuted
    tensor exactly matches the target parameter shape.

    The caller must have verified the checkpoint hash before calling this
    function.  The original checkpoint file is never modified.
    """

    converted = []
    unresolved = []
    for key, source in state_dict.items():
        target = model_state_dict.get(key)
        if target is None or tuple(source.shape) == tuple(target.shape):
            continue
        if key.startswith("pts_middle_encoder.") and source.ndim == 5:
            candidate = source.permute(1, 2, 3, 4, 0).contiguous()
            if tuple(candidate.shape) == tuple(target.shape):
                state_dict[key] = candidate
                converted.append(key)
                continue
        unresolved.append(
            f"{key}: checkpoint={tuple(source.shape)} model={tuple(target.shape)}"
        )
    if unresolved:
        preview = "\n  ".join(unresolved[:12])
        suffix = "" if len(unresolved) <= 12 else f"\n  ... 另有 {len(unresolved) - 12} 项"
        raise ValueError(f"checkpoint 仍有无法安全转换的参数形状：\n  {preview}{suffix}")
    return converted


def _build_model_with_compatible_checkpoint(
    model_config, checkpoint_payload: dict, device: str, torch
):
    """Build the official model and load a fully shape-checked state dict."""

    import warnings

    from mmengine.registry import init_default_scope
    from mmdet3d.apis.inference import convert_SyncBN
    from mmdet3d.registry import DATASETS, MODELS
    from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

    convert_SyncBN(model_config.model)
    model_config.model.train_cfg = None
    init_default_scope(model_config.get("default_scope", "mmdet3d"))
    model = MODELS.build(model_config.model)

    state_dict = checkpoint_payload.get("state_dict", checkpoint_payload)
    converted = _convert_sparseconv_checkpoint_layout(
        state_dict, model.state_dict()
    )
    # strict=True is intentional: a zero-detection result is not acceptable if
    # any official detector parameter was silently skipped.
    model.load_state_dict(state_dict, strict=True)

    meta = checkpoint_payload.get("meta", {})
    if "dataset_meta" in meta:
        model.dataset_meta = meta["dataset_meta"]
    elif "CLASSES" in meta:
        model.dataset_meta = {"classes": meta["CLASSES"]}
        if "PALETTE" in meta:
            model.dataset_meta["palette"] = meta["PALETTE"]
    else:
        model.dataset_meta = {"classes": model_config.class_names}

    test_dataset_cfg = deepcopy(model_config.test_dataloader.dataset)
    test_dataset_cfg["lazy_init"] = True
    cfg_palette = DATASETS.build(test_dataset_cfg).metainfo.get("palette")
    if cfg_palette is not None:
        model.dataset_meta["palette"] = cfg_palette
    elif "palette" not in model.dataset_meta:
        warnings.warn("palette does not exist; random palette is used")
        model.dataset_meta["palette"] = "random"

    model.cfg = model_config
    if device != "cpu":
        torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    backend = "spconv2" if IS_SPCONV2_AVAILABLE else "mmcv_sparseconv"
    return model, backend, converted


def _verify_checkpoint_sha256(checkpoint_path: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest().upper()
    expected = expected_sha256.strip().upper()
    if actual != expected:
        raise ValueError(
            f"checkpoint SHA256不匹配；actual={actual} expected={expected}"
        )
    return actual


def _load_backend_model(args, mmengine, torch):
    """Load the trusted official checkpoint once for frame or sequence inference."""

    checkpoint_path = Path(args.checkpoint).resolve()
    verified_sha256 = _verify_checkpoint_sha256(
        checkpoint_path, args.expected_checkpoint_sha256
    )
    # PyTorch>=2.6 defaults to weights_only=True.  This pinned official
    # checkpoint contains MMEngine HistoryBuffer metadata, so legacy pickle is
    # enabled only after exact SHA256 verification.
    checkpoint_payload = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    model_config = mmengine.Config.fromfile(str(Path(args.config).resolve()))
    redundant_source = _disable_redundant_backbone_init(
        model_config, checkpoint_payload, torch
    )
    if redundant_source:
        print("[模型] 完整checkpoint已含Swin权重，跳过冗余预训练下载")
    model_started = time.perf_counter()
    model, sparseconv_backend, converted_sparseconv_keys = (
        _build_model_with_compatible_checkpoint(
            model_config, checkpoint_payload, args.device, torch
        )
    )
    del checkpoint_payload
    model_load_latency = time.perf_counter() - model_started
    print(
        f"[模型兼容] backend={sparseconv_backend} | "
        f"SparseConv权重转换={len(converted_sparseconv_keys)} | strict_load=通过"
    )
    return model, {
        "checkpoint_sha256": verified_sha256,
        "model_load_latency_s": model_load_latency,
        "redundant_backbone_init_disabled": bool(redundant_source),
        "sparseconv_backend": sparseconv_backend,
        "sparseconv_layout_converted_count": len(converted_sparseconv_keys),
        "checkpoint_strict_load": True,
    }


def _prediction_detections(prediction, np, score_threshold: float) -> list[dict]:
    instances = prediction.pred_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
    scores = instances.scores_3d.detach().cpu().numpy()
    labels = instances.labels_3d.detach().cpu().numpy().astype(int)
    keep = np.flatnonzero(scores >= score_threshold)
    detections = []
    for index in keep:
        row = boxes[index]
        detections.append(
            {
                "center_lidar_m": row[:3].astype(float).tolist(),
                # MMDetection3D LiDARInstance3DBoxes: x_size/y_size/z_size.
                "size_lwh_m": row[3:6].astype(float).tolist(),
                "yaw_lidar_rad": float(row[6]),
                "class_name": NUSCENES_CLASSES[int(labels[index])],
                "score": float(scores[index]),
                "velocity_lidar_mps": (
                    row[7:9].astype(float).tolist() if len(row) >= 9 else None
                ),
            }
        )
    return detections


def _predict_prepared(
    model,
    prepared,
    pseudo_collate,
    torch,
    device: str,
    *,
    amp_fp16: bool = False,
):
    started = time.perf_counter()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    with torch.inference_mode(), torch.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=bool(amp_fp16 and device_type == "cuda"),
    ):
        prediction = model.test_step(pseudo_collate([prepared]))[0]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return prediction, time.perf_counter() - started


def infer(args: argparse.Namespace) -> int:
    repo_root = _activate_repo(args.repo_root)
    mmengine, np, torch, Compose, pseudo_collate, get_box_type = _imports()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    info_files = [Path(path).resolve() for path in args.info_file]
    data_root = Path(args.data_root).resolve()
    data_info = _absolute_data_paths(
        _select_data_info(mmengine, info_files, request["sample_token"]), data_root
    )

    model, model_metadata = _load_backend_model(args, mmengine, torch)

    pipeline = Compose(deepcopy(model.cfg.test_dataloader.dataset.pipeline))
    box_type_3d, box_mode_3d = get_box_type(
        model.cfg.test_dataloader.dataset.box_type_3d
    )
    data_info["box_type_3d"] = box_type_3d
    data_info["box_mode_3d"] = box_mode_3d
    prepared = pipeline(data_info)

    # 首次前向包含 kernel 初始化；记录但不把模型加载时间混入推理延迟。
    prediction, inference_latency = _predict_prepared(
        model, prepared, pseudo_collate, torch, args.device
    )
    detections = _prediction_detections(prediction, np, args.score_threshold)

    output = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_frame": "LIDAR_TOP",
        "sample_token": request["sample_token"],
        "timestamp_s": float(request["timestamp_s"]),
        "backend": "mmdetection3d.projects.BEVFusion",
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_sha256": model_metadata["checkpoint_sha256"],
        "inference_latency_s": inference_latency,
        "model_load_latency_s": model_metadata["model_load_latency_s"],
        "repo_commit": os.popen(f'git -C "{repo_root}" rev-parse HEAD').read().strip(),
        **model_metadata,
        "detections": detections,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[BEVFusion] detections={len(detections)} | "
        f"model_load={model_metadata['model_load_latency_s']:.2f}s | "
        f"inference={inference_latency:.3f}s"
    )
    print(f"[输出] {destination}")
    return 0


def infer_sequence(args: argparse.Namespace) -> int:
    """Run ordered frames with one persistent model instance on the GPU."""

    repo_root = _activate_repo(args.repo_root)
    mmengine, np, torch, Compose, pseudo_collate, get_box_type = _imports()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if request.get("schema_version") != SEQUENCE_SCHEMA_VERSION:
        raise ValueError("infer-sequence请求的schema_version不兼容")
    requested_frames = request.get("frames", [])
    if len(requested_frames) < 2:
        raise ValueError("infer-sequence至少需要2帧")
    timestamps = [float(item["timestamp_s"]) for item in requested_frames]
    if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
        raise ValueError("infer-sequence请求时间戳必须严格递增")

    info_files = [Path(path).resolve() for path in args.info_file]
    data_root = Path(args.data_root).resolve()
    info_by_token = _index_data_infos(mmengine, info_files)
    missing = [
        item["sample_token"]
        for item in requested_frames
        if item["sample_token"] not in info_by_token
    ]
    if missing:
        raise ValueError(f"sequence中有{len(missing)}个sample_token不在info文件")

    model, model_metadata = _load_backend_model(args, mmengine, torch)
    pipeline = Compose(deepcopy(model.cfg.test_dataloader.dataset.pipeline))
    box_type_3d, box_mode_3d = get_box_type(
        model.cfg.test_dataloader.dataset.box_type_3d
    )
    frame_outputs = []
    for frame_index, requested in enumerate(requested_frames, start=1):
        token = str(requested["sample_token"])
        data_info = _absolute_data_paths(info_by_token[token], data_root)
        data_info["box_type_3d"] = box_type_3d
        data_info["box_mode_3d"] = box_mode_3d
        prepared = pipeline(data_info)
        prediction, latency_s = _predict_prepared(
            model, prepared, pseudo_collate, torch, args.device
        )
        detections = _prediction_detections(
            prediction, np, args.score_threshold
        )
        frame_outputs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "coordinate_frame": "LIDAR_TOP",
                "sample_token": token,
                "timestamp_s": float(requested["timestamp_s"]),
                "backend": "mmdetection3d.projects.BEVFusion",
                "checkpoint": Path(args.checkpoint).name,
                "inference_latency_s": latency_s,
                "detections": detections,
            }
        )
        print(
            f"[序列推理 {frame_index:02d}/{len(requested_frames):02d}] "
            f"frame={requested.get('frame_offset')} | detections={len(detections)} | "
            f"latency={latency_s:.3f}s"
        )

    output = {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "scene_name": str(request["scene_name"]),
        "checkpoint": Path(args.checkpoint).name,
        "backend_score_threshold": float(args.score_threshold),
        "repo_commit": os.popen(f'git -C "{repo_root}" rev-parse HEAD').read().strip(),
        **model_metadata,
        "frames": frame_outputs,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latencies = [item["inference_latency_s"] for item in frame_outputs]
    steady = latencies[1:] if len(latencies) > 1 else latencies
    print(
        f"[BEVFusion序列] frames={len(frame_outputs)} | "
        f"first={latencies[0]:.3f}s | steady_mean={sum(steady) / len(steady):.3f}s"
    )
    print(f"[输出] {destination}")
    return 0


def _online_data_info(payload: dict, np) -> dict:
    """把Windows桥接层写出的纯JSON标定恢复为MMDet3D数据字典。

    在线CARLA帧不伪造nuScenes标注；这里只包含六相机、当前LiDAR和标定。
    ``LoadPointsFromMultiSweeps`` 会按官方配置用当前帧填充空sweeps。
    """

    data_info = payload["data_info"]
    images = {}
    for channel, camera in data_info["images"].items():
        images[channel] = {
            **camera,
            "cam2img": np.asarray(camera["cam2img"], dtype=np.float32),
            "lidar2cam": np.asarray(camera["lidar2cam"], dtype=np.float32),
        }
    return {
        **data_info,
        "ego2global": np.asarray(data_info["ego2global"], dtype=np.float32),
        "images": images,
        "lidar_points": {
            **data_info["lidar_points"],
            "lidar2ego": np.asarray(
                data_info["lidar_points"]["lidar2ego"], dtype=np.float32
            ),
        },
        "lidar_sweeps": [],
    }


def _online_pipeline_config(model) -> list:
    """CARLA桥已完成官方确定性图像增强时，将ImageAug3D改为恒等变换。"""

    pipeline_config = deepcopy(model.cfg.test_dataloader.dataset.pipeline)
    for transform in pipeline_config:
        transform_type = str(transform.get("type", ""))
        if transform_type.endswith("ImageAug3D"):
            transform["final_dim"] = [256, 704]
            transform["resize_lim"] = [1.0, 1.0]
            transform["bot_pct_lim"] = [0.0, 0.0]
            transform["rot_lim"] = [0.0, 0.0]
            transform["rand_flip"] = False
            transform["is_train"] = False
    return pipeline_config


def _run_pipeline_profiled(pipeline, data_info: dict) -> tuple[dict, dict[str, float]]:
    """Run MMEngine Compose unchanged while exposing per-transform wall time."""

    stage_latencies: dict[str, float] = {}
    prepared = data_info
    for index, transform in enumerate(pipeline.transforms):
        started = time.perf_counter()
        prepared = transform(prepared)
        elapsed = time.perf_counter() - started
        name = f"{index:02d}_{type(transform).__name__}"
        stage_latencies[name] = elapsed
        if prepared is None:
            raise RuntimeError(f"online BEVFusion pipeline rejected frame at {name}")
    return prepared, stage_latencies


class _OnlineIdentityImageAug3D:
    """Emit the augmentation matrices for already-canonical online images.

    The Windows bridge has already performed the exact official deterministic
    resize/crop and updated ``cam2img``.  Re-running PIL resize/crop with an
    identity configuration cost about 17 ms per frame and could only add a
    second interpolation.  This transform preserves the fields consumed by
    BEVFusion while leaving image pixels untouched.
    """

    def __init__(self, np_module):
        self._np = np_module

    def __call__(self, data: dict) -> dict:
        images = data["img"]
        data["img_aug_matrix"] = [
            self._np.eye(4, dtype=self._np.float32) for _ in images
        ]
        return data


def _replace_online_identity_image_aug(pipeline, np_module) -> bool:
    """Replace only ImageAug3D in a built online pipeline."""

    replaced = False
    for index, transform in enumerate(pipeline.transforms):
        if type(transform).__name__ == "ImageAug3D":
            pipeline.transforms[index] = _OnlineIdentityImageAug3D(np_module)
            replaced = True
    return replaced


class _OnlineInlineImageLoader:
    """Decode the bridge's in-request JPEGs with official loader semantics.

    The source bytes are produced by the same OpenCV JPEG encoder as the file
    transport and decoded by the same MMEngine/Pillow path.  Only transport is
    changed; image order, RGB convention, calibration and tensor packing stay
    identical to ``BEVLoadMultiViewImageFromFiles``.
    """

    def __init__(self, original, np_module, mmcv_module):
        self._np = np_module
        self._mmcv = mmcv_module
        self.to_float32 = bool(original.to_float32)
        self.color_type = original.color_type
        self.set_default_scale = bool(original.set_default_scale)
        self.num_views = int(original.num_views)
        self.num_ref_frames = int(original.num_ref_frames)
        if self.num_ref_frames > 0:
            raise ValueError("inline CARLA transport supports current-frame views only")

    def __call__(self, results: dict) -> dict:
        np = self._np
        cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], []
        images = []
        filenames = []
        for channel, camera in results["images"].items():
            lidar2cam_array = np.asarray(
                camera["lidar2cam"], dtype=np.float32
            )
            lidar2cam.append(lidar2cam_array)
            rotation = lidar2cam_array[:3, :3]
            translation = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4, dtype=np.float32)
            camera2lidar[:3, :3] = rotation.T
            camera2lidar[:3, 3:4] = -rotation.T @ translation.reshape(3, 1)
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4, dtype=np.float32)
            cam2img_array[:3, :3] = np.asarray(
                camera["cam2img"], dtype=np.float32
            )
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

            encoded = base64.b64decode(camera["img_bytes_b64"], validate=True)
            images.append(self._mmcv.imfrombytes(
                encoded,
                flag=self.color_type,
                backend="pillow",
                channel_order="rgb",
            ))
            filenames.append(f"inline://{channel}")

        shapes = np.stack([image.shape for image in images], axis=0)
        shape_max = np.max(shapes, axis=0)
        shape_min = np.min(shapes, axis=0)
        if shape_min[-1] != shape_max[-1]:
            raise ValueError("inline camera channel counts differ")
        if not np.all(shape_max == shape_min):
            images = [
                self._mmcv.impad(image, shape=shape_max[:2], pad_val=0)
                for image in images
            ]
        stacked = np.stack(images, axis=-1)
        if self.to_float32:
            stacked = stacked.astype(np.float32)

        results["img_path"] = filenames
        results["filename"] = filenames
        results["cam2img"] = np.stack(cam2img, axis=0)
        results["lidar2cam"] = np.stack(lidar2cam, axis=0)
        results["cam2lidar"] = np.stack(cam2lidar, axis=0)
        results["lidar2img"] = np.stack(lidar2img, axis=0)
        results["ori_cam2img"] = deepcopy(results["cam2img"])
        results["img"] = [
            stacked[..., index] for index in range(stacked.shape[-1])
        ]
        results["img_shape"] = stacked.shape[:2]
        results["ori_shape"] = stacked.shape[:2]
        results["pad_shape"] = stacked.shape[:2]
        if self.set_default_scale:
            results["scale_factor"] = 1.0
        channels = 1 if len(stacked.shape) < 3 else stacked.shape[2]
        results["img_norm_cfg"] = {
            "mean": np.zeros(channels, dtype=np.float32),
            "std": np.ones(channels, dtype=np.float32),
            "to_rgb": False,
        }
        results["num_views"] = self.num_views
        results["num_ref_frames"] = self.num_ref_frames
        return results


def _replace_online_inline_image_loader(pipeline, np_module) -> bool:
    import mmcv

    for index, transform in enumerate(pipeline.transforms):
        if type(transform).__name__ == "BEVLoadMultiViewImageFromFiles":
            pipeline.transforms[index] = _OnlineInlineImageLoader(
                transform, np_module, mmcv
            )
            return True
    return False


def serve_carla(args: argparse.Namespace) -> int:
    """常驻GPU模型，通过stdin逐行接收CARLA单帧请求。

    控制消息格式为 ``{"request_path": ..., "output_path": ...}``；路径均为
    WSL绝对路径。每帧结果仍使用稳定的BEVFusion JSON契约，便于故障审计。
    """

    _activate_repo(args.repo_root)
    mmengine, np, torch, Compose, pseudo_collate, get_box_type = _imports()
    model, model_metadata = _load_backend_model(args, mmengine, torch)
    pipeline = Compose(_online_pipeline_config(model))
    identity_image_aug = _replace_online_identity_image_aug(pipeline, np)
    inline_image_loader = False
    if args.inline_images:
        inline_image_loader = _replace_online_inline_image_loader(pipeline, np)
        if not inline_image_loader:
            raise RuntimeError("BEVFusion online image loader was not found")
    box_type_3d, box_mode_3d = get_box_type(
        model.cfg.test_dataloader.dataset.box_type_3d
    )
    ready_path = Path(args.ready_file)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_temporary = ready_path.with_suffix(ready_path.suffix + ".tmp")
    ready_temporary.write_text(
        json.dumps(
            {
                **model_metadata,
                "online_identity_image_aug": identity_image_aug,
                "online_inline_image_loader": inline_image_loader,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ready_temporary.replace(ready_path)
    print("[BEVFusion CARLA服务] 模型已常驻，等待在线帧", flush=True)
    amp_active = bool(args.amp_fp16 and args.device.startswith("cuda"))
    amp_fallback_error = None

    for line in sys.stdin:
        if not line.strip():
            continue
        command = json.loads(line)
        if command.get("command") == "shutdown":
            print("[BEVFusion CARLA服务] 收到关闭请求", flush=True)
            return 0
        destination = Path(command["output_path"])
        backend_started = time.perf_counter()
        request_read_started = backend_started
        request_transport = "stdin" if "request" in command else "file"
        if request_transport == "stdin":
            request = command["request"]
        else:
            request_path = Path(command["request_path"])
            request = json.loads(request_path.read_text(encoding="utf-8"))
        request_read_latency = time.perf_counter() - request_read_started
        data_info_started = time.perf_counter()
        data_info = _online_data_info(request, np)
        data_info["box_type_3d"] = box_type_3d
        data_info["box_mode_3d"] = box_mode_3d
        data_info_latency = time.perf_counter() - data_info_started
        prepared_started = time.perf_counter()
        prepared, transform_latencies = _run_pipeline_profiled(pipeline, data_info)
        preprocess_latency = time.perf_counter() - prepared_started
        try:
            prediction, inference_latency = _predict_prepared(
                model, prepared, pseudo_collate, torch, args.device,
                amp_fp16=amp_active,
            )
        except RuntimeError as exc:
            if not amp_active:
                raise
            amp_fallback_error = f"{type(exc).__name__}: {exc}"
            amp_active = False
            torch.cuda.empty_cache()
            print(
                f"[BEVFusion AMP回退] FP16不兼容，后续改用FP32：{exc}",
                flush=True,
            )
            prediction, inference_latency = _predict_prepared(
                model, prepared, pseudo_collate, torch, args.device,
                amp_fp16=False,
            )
        postprocess_started = time.perf_counter()
        detections = _prediction_detections(
            prediction, np, args.score_threshold
        )
        postprocess_latency = time.perf_counter() - postprocess_started
        backend_total_latency = time.perf_counter() - backend_started
        output = {
            "schema_version": SCHEMA_VERSION,
            "coordinate_frame": "LIDAR_TOP",
            "sample_token": str(request["sample_token"]),
            "timestamp_s": float(request["timestamp_s"]),
            "backend": "mmdetection3d.projects.BEVFusion",
            "checkpoint": Path(args.checkpoint).name,
            "inference_latency_s": inference_latency,
            "backend_preprocess_latency_s": preprocess_latency,
            "inference_precision": "amp_fp16" if amp_active else "fp32",
            "amp_fallback_error": amp_fallback_error,
            "backend_stage_latencies_s": {
                "request_json_read": request_read_latency,
                "data_info_build": data_info_latency,
                **transform_latencies,
                "prediction_to_cpu_json": postprocess_latency,
                "backend_total_before_publish": backend_total_latency,
            },
            "request_transport": request_transport,
            "detections": detections,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(output, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(destination)
        print(
            f"[BEVFusion在线] token={request['sample_token']} | "
            f"detections={len(detections)} | inference={inference_latency:.3f}s",
            flush=True,
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="isolated MMDetection3D BEVFusion backend")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--repo-root", required=True)
    prepare = subparsers.add_parser("prepare-infos")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--prefix", default="fusion_tracking_nuscenes")
    run = subparsers.add_parser("infer")
    sequence = subparsers.add_parser("infer-sequence")
    for inference_parser in (run, sequence):
        inference_parser.add_argument("--repo-root", required=True)
        inference_parser.add_argument("--config", required=True)
        inference_parser.add_argument("--checkpoint", required=True)
        inference_parser.add_argument(
            "--expected-checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256
        )
        inference_parser.add_argument("--data-root", required=True)
        inference_parser.add_argument("--info-file", required=True, nargs="+")
        inference_parser.add_argument("--request", required=True)
        inference_parser.add_argument("--output", required=True)
        inference_parser.add_argument("--device", default="cuda:0")
        inference_parser.add_argument("--score-threshold", type=float, default=0.25)
    serve = subparsers.add_parser("serve-carla")
    serve.add_argument("--repo-root", required=True)
    serve.add_argument("--config", required=True)
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument(
        "--expected-checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256
    )
    serve.add_argument("--ready-file", required=True)
    serve.add_argument("--device", default="cuda:0")
    serve.add_argument("--score-threshold", type=float, default=0.10)
    serve.add_argument("--amp-fp16", action="store_true")
    serve.add_argument(
        "--inline-images",
        action="store_true",
        help="read six base64 JPEG streams from the atomic request JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        return preflight(args.repo_root)
    if args.command == "prepare-infos":
        return prepare_infos(args.repo_root, args.data_root, args.prefix)
    if args.command == "infer-sequence":
        return infer_sequence(args)
    if args.command == "serve-carla":
        return serve_carla(args)
    return infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
