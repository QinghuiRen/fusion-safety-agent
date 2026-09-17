import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fusion_tracking_bevfusion import (
    BEVFusionAdapterConfig,
    BEVFusionBox,
    BEVFusionFrameResult,
    BEVFusionSequenceResult,
    detections_from_bevfusion,
)
from fusion_tracking_bevfusion.backend_mmdet3d import (
    _OnlineIdentityImageAug3D,
    _absolute_data_paths,
    _convert_sparseconv_checkpoint_layout,
    _disable_redundant_backbone_init,
    _index_data_infos,
    _run_pipeline_profiled,
    _select_data_info,
    _verify_checkpoint_sha256,
)
from fusion_tracking_bevfusion.carla_online import (
    CarlaBEVFusionConfig,
    _check_wsl_runtime,
    _packet_roots,
    wsl_native_windows_path,
)
from fusion_tracking_carla.tracker import StreamingFusionTracker
from fusion_tracking_carla.types import RadarReturn


def _box(class_name="car", score=0.9):
    return BEVFusionBox(
        center_lidar_m=(1.0, 2.0, 0.0),
        size_lwh_m=(4.0, 2.0, 2.0),
        yaw_lidar_rad=0.0,
        class_name=class_name,
        score=score,
        velocity_lidar_mps=(3.0, 0.0),
    )


def _result(*boxes):
    return BEVFusionFrameResult(
        sample_token="sample-a",
        timestamp_s=12.5,
        detections=tuple(boxes),
        checkpoint="official.pth",
    )


def test_schema_round_trip_and_version_guard(tmp_path):
    path = _result(_box()).dump(tmp_path / "result.json")
    loaded = BEVFusionFrameResult.load(path)
    assert loaded == _result(_box())

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "future.v2"
    with pytest.raises(ValueError, match="schema_version"):
        BEVFusionFrameResult.from_dict(payload)


def test_frame_result_preserves_backend_timing_metadata():
    payload = _result(_box()).to_dict()
    payload.update({
        "backend_preprocess_latency_s": 0.081,
        "inference_latency_s": 0.423,
        "inference_precision": "amp_fp16",
        "amp_fallback_error": None,
        "backend_stage_latencies_s": {
            "00_BEVLoadMultiViewImageFromFiles": 0.05,
            "backend_total_before_publish": 0.50,
        },
    })
    parsed = BEVFusionFrameResult.from_dict(payload)
    assert parsed.backend_preprocess_latency_s == 0.081
    assert parsed.inference_latency_s == 0.423
    assert parsed.inference_precision == "amp_fp16"
    assert parsed.backend_stage_latencies_s[
        "00_BEVLoadMultiViewImageFromFiles"
    ] == 0.05


def test_profiled_online_pipeline_preserves_transform_order_and_result():
    class AddOne:
        def __call__(self, payload):
            return {**payload, "value": payload["value"] + 1}

    class Double:
        def __call__(self, payload):
            return {**payload, "value": payload["value"] * 2}

    prepared, timings = _run_pipeline_profiled(
        type("Pipeline", (), {"transforms": [AddOne(), Double()]})(),
        {"value": 2},
    )
    assert prepared["value"] == 6
    assert list(timings) == ["00_AddOne", "01_Double"]
    assert all(value >= 0.0 for value in timings.values())


def test_online_identity_image_aug_preserves_pixels_and_emits_matrices():
    images = [np.zeros((256, 704, 3), dtype=np.float32) for _ in range(6)]
    prepared = _OnlineIdentityImageAug3D(np)({"img": images})

    assert all(actual is expected for actual, expected in zip(prepared["img"], images))
    assert len(prepared["img_aug_matrix"]) == 6
    for matrix in prepared["img_aug_matrix"]:
        np.testing.assert_array_equal(matrix, np.eye(4, dtype=np.float32))


def test_wsl_native_packet_path_is_a_real_unc_anchor():
    path = wsl_native_windows_path("Ubuntu-22.04", "/home/example/packets/run")
    assert path.is_absolute()
    assert str(path).startswith("\\\\wsl.localhost\\Ubuntu-22.04\\")


def test_stdin_packet_transport_uses_local_ntfs_not_wsl_unc(tmp_path):
    config = CarlaBEVFusionConfig(
        packet_transport="wsl_stdin_packet",
        packet_root=tmp_path / "packets",
    )

    windows_root, backend_root = _packet_roots(config, "session_123")

    assert windows_root == (tmp_path / "packets" / "session_123").resolve()
    assert not str(windows_root).startswith("\\\\wsl")
    assert backend_root.startswith("/mnt/")
    assert backend_root.endswith("/packets/session_123")


def test_inline_image_transport_keeps_wsl_native_small_file_root():
    config = CarlaBEVFusionConfig(packet_transport="wsl_inline_images")

    windows_root, backend_root = _packet_roots(config, "session_123")

    assert str(windows_root).startswith("\\\\wsl.localhost\\Ubuntu-22.04\\")
    assert backend_root.endswith("/carla_packet_ipc/session_123")


def test_wsl_preflight_times_out_with_actionable_error(monkeypatch):
    def timeout(*args, **kwargs):
        raise __import__("subprocess").TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(
        "fusion_tracking_bevfusion.carla_online.subprocess.run", timeout
    )

    with pytest.raises(RuntimeError, match="restart WSL"):
        _check_wsl_runtime("Ubuntu-22.04", timeout_s=3.0)


def test_wsl_preflight_accepts_expected_handshake(monkeypatch):
    completed = __import__("types").SimpleNamespace(
        returncode=0, stdout="fusion-safety-wsl-ready", stderr=""
    )
    monkeypatch.setattr(
        "fusion_tracking_bevfusion.carla_online.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    assert _check_wsl_runtime("Ubuntu-22.04") >= 0.0


def test_wsl_runtime_home_is_resolved_without_storing_a_username(monkeypatch):
    from fusion_tracking_bevfusion.carla_online import resolve_wsl_runtime

    completed = __import__("types").SimpleNamespace(
        returncode=0, stdout="/home/runtime-user", stderr=""
    )
    monkeypatch.setattr(
        "fusion_tracking_bevfusion.carla_online.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    assert resolve_wsl_runtime(
        "Ubuntu-22.04", "~/fusion_tracking_runtime/bevfusion"
    ) == "/home/runtime-user/fusion_tracking_runtime/bevfusion"
    assert resolve_wsl_runtime(
        "Ubuntu-22.04", "/srv/fusion/bevfusion"
    ) == "/srv/fusion/bevfusion"


def test_adapter_transforms_counts_support_and_associates_radar():
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 20.0, 1.0]
    lidar = np.array([
        [1.0, 2.0, 0.0],
        [2.9, 2.9, 0.9],
        [3.1, 2.0, 0.0],
        [20.0, 20.0, 0.0],
    ])
    radar = RadarReturn(
        position_global=np.array([11.5, 22.0, 1.0]),
        sensor_origin_global=np.zeros(3),
        radial_unit_global=np.array([1.0, 0.0]),
        velocity_towards_mps=-3.0,
        sensor_name="RADAR_FRONT",
    )
    detections = detections_from_bevfusion(
        _result(_box()),
        transform,
        radar_returns=[radar],
        lidar_points_lidar=lidar,
    )
    assert len(detections) == 1
    detection = detections[0]
    np.testing.assert_allclose(detection.position_global, [11.0, 22.0])
    np.testing.assert_allclose(detection.extent_xy, [4.0, 2.0])
    np.testing.assert_allclose(detection.velocity_global, [3.0, 0.0])
    assert detection.velocity_source == "bevfusion_velocity_head"
    assert detection.lidar_support == 2
    assert detection.radar_returns == [radar]
    assert detection.family == "vehicle"


def test_adapter_can_disable_model_velocity_for_ablation():
    detection = detections_from_bevfusion(
        _result(_box()),
        np.eye(4),
        config=BEVFusionAdapterConfig(use_model_velocity=False),
    )[0]
    assert detection.velocity_global is None
    assert detection.velocity_std_mps is None
    assert detection.velocity_source is None


def test_adapter_filters_non_tracking_classes_and_low_scores():
    detections = detections_from_bevfusion(
        _result(_box("barrier", 0.99), _box("pedestrian", 0.2), _box("bicycle", 0.8)),
        np.eye(4),
        config=BEVFusionAdapterConfig(score_threshold=0.25),
    )
    assert [item.class_name for item in detections] == ["bicycle"]
    assert detections[0].family == "vru"


def test_adapter_assigns_each_radar_return_to_only_the_nearest_box():
    near = _box()
    far = BEVFusionBox(
        center_lidar_m=(6.0, 2.0, 0.0),
        size_lwh_m=(4.0, 2.0, 2.0),
        yaw_lidar_rad=0.0,
        class_name="car",
        score=0.9,
        velocity_lidar_mps=None,
    )
    radar = RadarReturn(
        position_global=np.array([1.4, 2.0, 0.0]),
        sensor_origin_global=np.zeros(3),
        radial_unit_global=np.array([1.0, 0.0]),
        velocity_towards_mps=-2.0,
        sensor_name="RADAR_FRONT",
    )
    detections = detections_from_bevfusion(
        _result(near, far), np.eye(4), radar_returns=[radar]
    )
    assert [len(item.radar_returns) for item in detections] == [1, 0]


def test_adapter_bev_nms_suppresses_overlapping_same_family_duplicates():
    primary = _box(score=0.9)
    duplicate = BEVFusionBox(
        center_lidar_m=(1.4, 2.1, 0.0),
        size_lwh_m=(4.2, 2.0, 2.0),
        yaw_lidar_rad=0.05,
        class_name="truck",
        score=0.7,
        velocity_lidar_mps=None,
    )
    adjacent = BEVFusionBox(
        center_lidar_m=(1.0, 5.6, 0.0),
        size_lwh_m=(4.0, 2.0, 2.0),
        yaw_lidar_rad=0.0,
        class_name="car",
        score=0.8,
        velocity_lidar_mps=None,
    )
    detections = detections_from_bevfusion(
        _result(primary, duplicate, adjacent), np.eye(4)
    )
    assert [item.score for item in detections] == [0.9, 0.8]


def test_bevfusion_detection_enters_existing_tracker_without_special_case():
    detections = detections_from_bevfusion(_result(_box()), np.eye(4))
    tracks = StreamingFusionTracker().update(
        detections, timestamp_s=12.5, ego_velocity_global=np.zeros(2)
    )
    assert len(tracks) == 1
    assert tracks[0].class_name == "car"
    np.testing.assert_allclose(tracks[0].state_global[:2], [1.0, 2.0])


def test_backend_restores_v2_sensor_prefixes(tmp_path):
    info = {
        "lidar_points": {"lidar_path": "key.bin"},
        "images": {
            "CAM_FRONT": {"img_path": "front.jpg"},
            "CAM_BACK": {"img_path": "back.jpg"},
        },
        "lidar_sweeps": [
            {"lidar_points": {"lidar_path": "sweep.bin"}},
        ],
    }
    converted = _absolute_data_paths(info, tmp_path)
    assert converted["lidar_points"]["lidar_path"].replace("\\", "/").endswith(
        "samples/LIDAR_TOP/key.bin"
    )
    assert converted["images"]["CAM_FRONT"]["img_path"].replace("\\", "/").endswith(
        "samples/CAM_FRONT/front.jpg"
    )
    assert converted["images"]["CAM_BACK"]["img_path"].replace("\\", "/").endswith(
        "samples/CAM_BACK/back.jpg"
    )
    sweep_path = converted["lidar_sweeps"][0]["lidar_points"]["lidar_path"]
    assert sweep_path.replace("\\", "/").endswith("sweeps/LIDAR_TOP/sweep.bin")


def test_backend_searches_train_and_val_by_v2_token():
    class FakeMMEngine:
        @staticmethod
        def load(path):
            if str(path).endswith("train.pkl"):
                return {"data_list": [{"sample_idx": 0, "token": "wanted"}]}
            return {"data_list": [{"sample_idx": 0, "token": "other"}]}

    selected = _select_data_info(
        FakeMMEngine, [Path("train.pkl"), Path("val.pkl")], "wanted"
    )
    assert selected["token"] == "wanted"


def test_backend_disables_redundant_swin_download_only_with_full_weights(tmp_path):
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    checkpoint = tmp_path / "full.pth"
    torch.save({"state_dict": {"img_backbone.block.weight": torch.ones(1)}}, checkpoint)
    model_config = AttrDict(
        model=AttrDict(
            img_backbone=AttrDict(
                init_cfg=AttrDict(checkpoint="https://example.invalid/swin.pth")
            )
        )
    )
    source = _disable_redundant_backbone_init(model_config, checkpoint, torch)
    assert source == "https://example.invalid/swin.pth"
    assert model_config.model.img_backbone.init_cfg is None

    incomplete = tmp_path / "incomplete.pth"
    torch.save({"state_dict": {"pts_backbone.weight": torch.ones(1)}}, incomplete)
    with pytest.raises(ValueError, match="img_backbone"):
        _disable_redundant_backbone_init(model_config, incomplete, torch)


def test_backend_requires_exact_checkpoint_hash_before_legacy_load(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"trusted-test-content")
    expected = "B83DA5F77E1240052A084AAD8950772F2B308E983A6775F15CD86F46A90C7CBA"
    assert _verify_checkpoint_sha256(checkpoint, expected.lower()) == expected
    with pytest.raises(ValueError, match="SHA256"):
        _verify_checkpoint_sha256(checkpoint, "0" * 64)


def test_backend_converts_only_shape_proven_sparseconv2_kernels():
    state_dict = {
        "pts_middle_encoder.conv.weight": torch.arange(16 * 3 * 3 * 3 * 5).reshape(
            16, 3, 3, 3, 5
        ),
        "img_backbone.block.weight": torch.ones(2, 2),
    }
    model_state = {
        "pts_middle_encoder.conv.weight": torch.empty(3, 3, 3, 5, 16),
        "img_backbone.block.weight": torch.empty(2, 2),
    }
    original = state_dict["pts_middle_encoder.conv.weight"].clone()

    converted = _convert_sparseconv_checkpoint_layout(state_dict, model_state)

    assert converted == ["pts_middle_encoder.conv.weight"]
    assert tuple(state_dict["pts_middle_encoder.conv.weight"].shape) == (
        3,
        3,
        3,
        5,
        16,
    )
    torch.testing.assert_close(
        state_dict["pts_middle_encoder.conv.weight"],
        original.permute(1, 2, 3, 4, 0),
    )


def test_backend_rejects_unproven_checkpoint_shape_mismatch():
    state_dict = {"img_backbone.block.weight": torch.ones(2, 3)}
    model_state = {"img_backbone.block.weight": torch.empty(3, 2)}
    with pytest.raises(ValueError, match="无法安全转换"):
        _convert_sparseconv_checkpoint_layout(state_dict, model_state)


def test_sequence_schema_round_trip_requires_strict_time_and_unique_tokens(tmp_path):
    first = _result(_box())
    second = BEVFusionFrameResult(
        sample_token="sample-b",
        timestamp_s=13.0,
        detections=(_box(),),
        checkpoint="official.pth",
    )
    sequence = BEVFusionSequenceResult(
        scene_name="scene-0061",
        frames=(first, second),
        checkpoint="official.pth",
        checkpoint_strict_load=True,
        backend_score_threshold=0.10,
    )
    path = sequence.dump(tmp_path / "sequence.json")
    assert BEVFusionSequenceResult.load(path) == sequence
    assert BEVFusionSequenceResult.load(path).backend_score_threshold == 0.10

    payload = sequence.to_dict()
    payload["frames"][1]["timestamp_s"] = first.timestamp_s
    with pytest.raises(ValueError, match="严格递增"):
        BEVFusionSequenceResult.from_dict(payload)


def test_backend_indexes_info_files_once_and_rejects_duplicate_tokens():
    class FakeMMEngine:
        payloads = {
            "train.pkl": {"data_list": [{"token": "train-a"}]},
            "val.pkl": {"data_list": [{"token": "val-a"}]},
            "duplicate.pkl": {"data_list": [{"token": "train-a"}]},
        }

        @classmethod
        def load(cls, path):
            return cls.payloads[str(path)]

    indexed = _index_data_infos(
        FakeMMEngine, [Path("train.pkl"), Path("val.pkl")]
    )
    assert set(indexed) == {"train-a", "val-a"}
    with pytest.raises(ValueError, match="重复"):
        _index_data_infos(
            FakeMMEngine, [Path("train.pkl"), Path("duplicate.pkl")]
        )
