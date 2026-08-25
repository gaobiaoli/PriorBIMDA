from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bim_priorda3.config import Config
from bim_priorda3.data.pano_tangent import (
    build_pano_tangent_preset,
    tangent_z_to_erp_range,
)
from bim_priorda3.data.stanford_pano import StanfordPanorama
from scripts.model.evaluate_stanford_pano import (
    DEPTH_METHODS,
    TANGENT_FUSION_METHODS,
    TANGENT_MANIFEST_PROTOCOL,
    TANGENT_MANIFEST_SCHEMA_VERSION,
    TANGENT_VARIANTS,
    ProjectedView,
    TangentPrepared,
    _aggregate_projected_views,
    _evaluate_regular_pano_joint,
    _evaluate_tangent_only_station,
    _fixed_pano_support,
    _geometry_fingerprint,
    _latitude_area_weights,
    _load_tangent_projected_views,
    _metrics_for_array,
    _paired_bootstrap,
    _per_room_rows,
    _photo_consistency_weight,
    _prepare_regular_pano_joint,
    _prepare_tangent_predictions,
    _room_macro_metrics,
    _route_paired_contrast,
    _select_strict_single_view,
    _sha256,
    _sorted_unique_intersection_positions,
    _station_bim_scale,
    _strict_comparison_predictions,
    _strict_fixed_support,
    _validate_tangent_manifest,
    _view_geometry_payload,
    _write_json,
    parse_args,
)


def _view(
    frame_id: str,
    indices: list[int],
    values: list[float],
    *,
    weights: list[float] | None = None,
    photo_weights: list[float] | None = None,
) -> ProjectedView:
    size = len(indices)
    base = np.ones(size, dtype=np.float32) if weights is None else np.asarray(weights, np.float32)
    photo = (
        np.ones(size, dtype=np.float32)
        if photo_weights is None
        else np.asarray(photo_weights, np.float32)
    )
    logs = np.log(np.asarray(values, dtype=np.float32))
    return ProjectedView(
        frame_id=frame_id,
        indices=np.asarray(indices, dtype=np.int64),
        base_weights=base,
        photo_weights=photo,
        log_ranges={"raw_da3": logs},
    )


def _synthetic_tangent_manifest(
    tmp_path,
    *,
    split: str = "val",
    pano_only: bool = True,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    cfg = Config(
        config_path=str(config_path),
        project_root=str(tmp_path),
        data=Config(
            split_annotation=str(annotation_path),
            da3_model="synthetic/da3",
            da3_revision="0123456789abcdef",
            da3_process_res=4,
            local_files_only=False,
            min_depth=0.2,
            max_depth=5.0,
        ),
    )
    split_provenance = {"mode": "annotations", "split": split, "fixture": "route-p-v3"}
    camera_uuid = "synthetic_station"
    room = "synthetic_room"
    dataset = SimpleNamespace(
        split_provenance=split_provenance,
        records=[] if pano_only else [{"camera_uuid": camera_uuid, "region": room}],
    )
    pano_rgb_path = tmp_path / "pano_rgb.png"
    pano_rgb = np.full((8, 16, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(pano_rgb_path), pano_rgb)
    pano_depth_path = tmp_path / "pano_depth.png"
    pano_depth = np.full((8, 16), 2 * 512, dtype=np.uint16)
    assert cv2.imwrite(str(pano_depth_path), pano_depth)
    semantic_path = tmp_path / "semantic.png"
    assert cv2.imwrite(str(semantic_path), np.zeros((8, 16), dtype=np.uint8))
    pose_path = tmp_path / "pose.json"
    pose_path.write_text("{}\n", encoding="utf-8")
    station = StanfordPanorama(
        key=f"{room}/{camera_uuid}",
        sample_id=f"pano/{camera_uuid}",
        room=room,
        pose_room=room,
        camera_uuid=camera_uuid,
        rgb_path=pano_rgb_path,
        depth_path=pano_depth_path,
        semantic_path=semantic_path,
        pose_path=pose_path,
        intrinsic=np.eye(3, dtype=np.float64),
        world_to_camera=np.eye(4, dtype=np.float64),
        camera_to_area=np.eye(4, dtype=np.float64),
        regular_views=() if pano_only else (object(),),
    )
    views = build_pano_tangent_preset("nested14", 4)
    geometry = [_view_geometry_payload(view, index) for index, view in enumerate(views)]
    fingerprint = _geometry_fingerprint(geometry)
    pano_sha = _sha256(pano_rgb_path)
    tangent_records = []
    for index, view in enumerate(views):
        tangent_path = tmp_path / f"tangent_{index:02d}.png"
        tangent_rgb = np.full((*view.image_shape, 3), 20 + index, dtype=np.uint8)
        assert cv2.imwrite(str(tangent_path), tangent_rgb)
        tangent_sha = _sha256(tangent_path)
        cache_path = tmp_path / f"tangent_{index:02d}.npz"
        np.savez_compressed(
            cache_path,
            schema_version=np.asarray(2, dtype=np.uint16),
            depth=np.full(view.image_shape, 2.0 + index * 0.01, dtype=np.float16),
            confidence=np.full(view.image_shape, 0.8, dtype=np.float16),
            image_sha256=np.asarray(tangent_sha),
            model_name=np.asarray("synthetic/da3"),
            model_revision=np.asarray("0123456789abcdef"),
            process_res=np.asarray(4, dtype=np.int32),
            target_shape=np.asarray(view.image_shape, dtype=np.int32),
            provenance_status=np.asarray("direct_inference"),
            local_files_only=np.asarray(False, dtype=np.bool_),
        )
        tangent_records.append(
            {
                "view": geometry[index],
                "pano_rgb_sha256": pano_sha,
                "tangent_rgb": {"path": str(tangent_path), "sha256": tangent_sha},
                "da3_cache": {
                    "path": str(cache_path),
                    "sha256": _sha256(cache_path),
                    "image_sha256": tangent_sha,
                    "model_name": "synthetic/da3",
                    "model_revision": "0123456789abcdef",
                    "process_res": 4,
                    "target_shape": [4, 4],
                    "provenance_status": "direct_inference",
                    "depth_quantity": "perspective_z_depth_m",
                },
            }
        )
    manifest = {
        "schema_version": TANGENT_MANIFEST_SCHEMA_VERSION,
        "protocol": TANGENT_MANIFEST_PROTOCOL,
        "split": split,
        "test_access_explicitly_authorized": split == "test",
        "preset": {
            "name": "nested14",
            "view_count": 14,
            "geometry_fingerprint_sha256": fingerprint,
            "cache_namespace": f"nested14_r4_{fingerprint[:12]}",
            "views": geometry,
        },
        "selection": {
            "room_source": "configured exhaustive split_annotation via BIMDepthDataset",
            "rooms": [room],
            "annotation_regular_station_count": 0 if pano_only else 1,
            "split_pano_station_count": 1,
            "shared_regular_pano_station_count": 0 if pano_only else 1,
            "pano_only_station_ids": [camera_uuid] if pano_only else [],
            "selected_station_count": 1,
            "selected_station_ids": [camera_uuid],
            "max_stations": None,
            "formal_protocol_eligible": True,
            "full_split_station_count_before_exploratory_limit": 1,
        },
        "split_provenance": split_provenance,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "split_annotation": {
            "path": str(annotation_path),
            "sha256": _sha256(annotation_path),
        },
        "model": {
            "name": "synthetic/da3",
            "revision": "0123456789abcdef",
            "process_res": 4,
            "local_files_only": False,
        },
        "input_contract": {
            "pano_rgb_decoded": True,
            "pano_pose_metadata_decoded": True,
            "pano_depth_decoded": False,
            "pano_semantic_decoded": False,
            "regular_ground_truth_decoded": False,
            "output_depth_quantity": "perspective_z_depth_m",
        },
        "stations": [
            {
                "camera_uuid": camera_uuid,
                "room": room,
                "pano_rgb": {"path": str(pano_rgb_path), "sha256": pano_sha},
                "tangent_views": tangent_records,
            }
        ],
    }
    manifest_path = tmp_path / "manifests" / f"{split}_full.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(
        path=manifest_path,
        payload=manifest,
        cfg=cfg,
        dataset=dataset,
        station=station,
    )


def test_two_pass_huber_rejects_one_inconsistent_view() -> None:
    truth = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    views = [
        _view("good_0", [0, 1, 2, 3], truth.tolist()),
        _view("good_1", [0, 1, 2, 3], truth.tolist()),
        _view("good_2", [0, 1, 2, 3], truth.tolist()),
        _view("outlier", [0, 1, 2, 3], (truth * math.exp(1.0)).tolist()),
    ]

    result = _aggregate_projected_views(
        views,
        ["raw_da3"],
        (2, 2),
        huber_delta=0.08,
        consistency_threshold=0.5,
        sync_min_overlap=1,
        sync_pair_max_samples=16,
    )

    weighted = result.predictions["joint_weighted_log"]["raw_da3"].reshape(-1)
    robust = result.predictions["joint_huber"]["raw_da3"].reshape(-1)
    assert np.mean(np.abs(np.log(robust / truth))) < np.mean(np.abs(np.log(weighted / truth)))
    assert np.allclose(robust, truth, rtol=1e-6, atol=1e-6)


def test_overlap_log_scale_synchronization_removes_relative_view_offsets() -> None:
    truth = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8], dtype=np.float32)
    views = [
        _view("low", [0, 1, 2], (truth[:3] * math.exp(-0.2)).tolist()),
        _view("middle", [0, 1, 2, 3, 4], truth.tolist()),
        _view("high", [2, 3, 4], (truth[2:] * math.exp(0.2)).tolist()),
    ]

    first = _aggregate_projected_views(
        views,
        ["raw_da3"],
        (1, 5),
        huber_delta=0.08,
        consistency_threshold=0.5,
        sync_min_overlap=2,
        sync_pair_max_samples=32,
    )
    second = _aggregate_projected_views(
        views,
        ["raw_da3"],
        (1, 5),
        huber_delta=0.08,
        consistency_threshold=0.5,
        sync_min_overlap=2,
        sync_pair_max_samples=32,
    )

    plain = first.predictions["joint_huber"]["raw_da3"].reshape(-1)
    synchronized = first.predictions["joint_synchronized_huber"]["raw_da3"].reshape(-1)
    assert np.mean(np.abs(np.log(synchronized / truth))) < np.mean(np.abs(np.log(plain / truth)))
    assert np.allclose(synchronized, truth, rtol=2e-5, atol=2e-5)
    offsets = first.synchronization["raw_da3"]["offsets_by_frame"]
    assert offsets["low"] == pytest.approx(0.2, abs=2e-5)
    assert offsets["middle"] == pytest.approx(0.0, abs=2e-5)
    assert offsets["high"] == pytest.approx(-0.2, abs=2e-5)
    assert np.array_equal(
        first.predictions["joint_synchronized_huber"]["raw_da3"],
        second.predictions["joint_synchronized_huber"]["raw_da3"],
    )


def test_synchronization_fixed_anchor_gauge_is_stable_when_views_are_appended() -> None:
    truth = np.asarray([1.0, 1.2, 1.4, 1.6], dtype=np.float32)
    anchors = [
        _view(
            "anchor_low",
            [0, 1, 2, 3],
            (truth * math.exp(-0.2)).tolist(),
            weights=[2.0] * 4,
        ),
        _view(
            "anchor_high",
            [0, 1, 2, 3],
            (truth * math.exp(0.2)).tolist(),
            weights=[1.0] * 4,
        ),
    ]
    appended = _view("new", [0, 1, 2, 3], (truth * math.exp(0.4)).tolist())
    base = _aggregate_projected_views(
        anchors,
        ["raw_da3"],
        (2, 2),
        huber_delta=0.08,
        consistency_threshold=0.5,
        sync_min_overlap=1,
        sync_pair_max_samples=16,
        sync_gauge_view_count=2,
    )
    extended = _aggregate_projected_views(
        [*anchors, appended],
        ["raw_da3"],
        (2, 2),
        huber_delta=0.08,
        consistency_threshold=0.5,
        sync_min_overlap=1,
        sync_pair_max_samples=16,
        sync_gauge_view_count=2,
    )

    base_offsets = base.synchronization["raw_da3"]["offsets_by_frame"]
    extended_offsets = extended.synchronization["raw_da3"]["offsets_by_frame"]
    assert extended.synchronization["raw_da3"]["gauge"]["anchor_view_count"] == 2
    assert extended.synchronization["raw_da3"]["gauge"][
        "normalized_anchor_weights"
    ] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert extended.synchronization["raw_da3"]["gauge"]["anchor_mean_offset"] == pytest.approx(
        0.0, abs=1e-8
    )
    assert extended_offsets["anchor_low"] == pytest.approx(base_offsets["anchor_low"], abs=1e-5)
    assert extended_offsets["anchor_high"] == pytest.approx(base_offsets["anchor_high"], abs=1e-5)


def test_sorted_intersection_large_arrays_is_exact_and_never_calls_intersect1d(
    monkeypatch,
) -> None:
    left = np.arange(0, 600_000, 2, dtype=np.int64)
    right = np.arange(0, 600_000, 3, dtype=np.int64)

    def unexpected_intersect(*args, **kwargs):
        raise AssertionError("optimized pair matching must not call np.intersect1d")

    monkeypatch.setattr(np, "intersect1d", unexpected_intersect)
    left_positions, right_positions = _sorted_unique_intersection_positions(left, right)
    expected = np.arange(0, 600_000, 6, dtype=np.int64)

    assert np.array_equal(left[left_positions], expected)
    assert np.array_equal(right[right_positions], expected)
    assert np.array_equal(left_positions, expected // 2)
    assert np.array_equal(right_positions, expected // 3)
    result = _aggregate_projected_views(
        [
            _view("left", [0, 1, 2], [1.0, 2.0, 3.0]),
            _view("right", [1, 2, 3], [2.0, 3.0, 4.0]),
        ],
        ["raw_da3"],
        (1, 4),
        huber_delta=0.08,
        consistency_threshold=0.5,
        sync_min_overlap=1,
        sync_pair_max_samples=16,
    )
    assert result.synchronization["raw_da3"]["pair_count"] == 1


def test_strict_single_selects_one_whole_frame_and_does_not_use_union_support() -> None:
    higher_id = _view("frame_z", [0, 1], [1.0, 2.0], weights=[1.0, 1.0])
    stable_tie_winner = _view("frame_a", [2, 3], [3.0, 4.0], weights=[1.0, 1.0])
    selected, score = _select_strict_single_view([higher_id, stable_tie_winner])
    joint = {
        method: {"raw_da3": np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)}
        for method in (
            "joint_weighted_log",
            "joint_huber",
            "joint_photo_huber",
            "joint_synchronized_huber",
        )
    }
    predictions = _strict_comparison_predictions(
        selected,
        joint,
        ["raw_da3"],
        (1, 4),
    )
    gt = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    support = _strict_fixed_support(
        gt,
        np.ones_like(gt, dtype=bool),
        selected,
        predictions,
    )

    assert selected.frame_id == "frame_a"
    assert score == pytest.approx(2.0)
    assert np.array_equal(support, np.asarray([[False, False, True, True]]))
    assert np.isnan(predictions["strict_single_frame"]["raw_da3"][0, :2]).all()


def test_room_rows_average_stations_before_room_macro() -> None:
    rows = []
    for station, room, value in (("a", "room_1", 0.1), ("b", "room_1", 0.3), ("c", "room_2", 0.6)):
        row = {
            "station_id": station,
            "room": room,
            "fusion_method": "joint_huber",
            "depth_method": "raw_da3",
            "fixed_support_pixels": 10,
        }
        row.update(
            {
                f"spherical_{name}": value
                for name in (
                    "abs_rel",
                    "mae",
                    "rmse",
                    "delta1",
                    "delta2",
                    "delta3",
                )
            }
        )
        rows.append(row)

    room_rows = _per_room_rows(rows)
    macro = _room_macro_metrics(room_rows, "raw_da3", "joint_huber")

    room_1 = next(row for row in room_rows if row["room"] == "room_1")
    assert room_1["spherical_abs_rel"] == pytest.approx(0.2)
    assert room_1["station_count"] == 2
    assert macro["spherical_abs_rel"] == pytest.approx(0.4)
    assert macro["room_count"] == 2
    assert macro["station_count"] == 3


def test_photometric_method_downweights_rgb_inconsistent_outlier() -> None:
    truth = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    views = [
        _view("good", [0, 1, 2, 3], truth.tolist()),
        _view(
            "bad_rgb",
            [0, 1, 2, 3],
            (truth * 1.5).tolist(),
            photo_weights=[1e-3] * 4,
        ),
    ]
    result = _aggregate_projected_views(
        views,
        ["raw_da3"],
        (2, 2),
        huber_delta=0.08,
        consistency_threshold=1.0,
        sync_min_overlap=1,
        sync_pair_max_samples=16,
    )

    plain = result.predictions["joint_huber"]["raw_da3"].reshape(-1)
    photo = result.predictions["joint_photo_huber"]["raw_da3"].reshape(-1)
    assert np.mean(np.abs(photo - truth)) < np.mean(np.abs(plain - truth))


def test_exposure_normalized_photo_weight_uses_only_rgb() -> None:
    projected = np.full((8, 8, 3), 0.25, dtype=np.float32)
    pano = np.full((8, 8, 3), 0.50, dtype=np.float32)
    valid = np.ones((8, 8), dtype=bool)

    weight, gain = _photo_consistency_weight(projected, pano, valid, sigma=0.12)
    mismatched, _ = _photo_consistency_weight(
        np.dstack((projected[..., 0], projected[..., 1] * 0.1, projected[..., 2])),
        pano,
        valid,
        sigma=0.12,
    )

    assert gain == pytest.approx(2.0, abs=0.01)
    assert float(np.min(weight)) > 0.99
    assert float(mismatched.mean()) < float(weight.mean())


def test_fixed_support_is_geometry_and_gt_defined_not_method_specific() -> None:
    gt = np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    gt_valid = gt > 0
    contributors = np.asarray([[1, 2, 0], [0, 1, 3]], dtype=np.int32)
    prediction = np.asarray([[1.0, 2.2, np.nan], [np.nan, 4.0, 5.5]], dtype=np.float32)
    predictions = {
        fusion: {method: prediction.copy() for method in DEPTH_METHODS}
        for fusion in (
            "single_best_view",
            "joint_weighted_log",
        )
    }

    support = _fixed_pano_support(gt, gt_valid, contributors, predictions)

    assert np.array_equal(
        support,
        np.asarray([[True, True, False], [False, True, True]]),
    )
    metrics = _metrics_for_array(prediction, gt, support)
    assert metrics["count"] == 4


def test_exact_erp_solid_angle_weights_integrate_to_full_sphere() -> None:
    weights = _latitude_area_weights(8, 16)

    assert weights.shape == (8, 16)
    assert float(weights.sum()) == pytest.approx(4.0 * math.pi)
    assert weights[0, 0] < weights[3, 0]


def test_json_writer_never_emits_nonstandard_nan_tokens(tmp_path) -> None:
    path = tmp_path / "result.json"

    _write_json(path, {"nan": float("nan"), "positive_infinity": float("inf")})

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {"nan": None, "positive_infinity": "inf"}


def test_station_scale_equal_view_sampling_is_deterministic(tmp_path) -> None:
    records = []
    for frame_number, ratio in enumerate((1.5, 2.0)):
        path = tmp_path / f"frame_{frame_number}.npz"
        np.savez_compressed(
            path,
            base_depth=np.ones((12, 12), dtype=np.float32),
            bim_depth=np.full((12, 12), ratio, dtype=np.float32),
            bim_valid=np.ones((12, 12), dtype=np.uint8),
        )
        records.append(
            {"id": f"room/frame_{frame_number}", "frame_number": frame_number, "sample": path}
        )
    cfg = Config(
        model=Config(
            scale_estimator=Config(
                name="log_upper_cap_v1",
                q10_log_cap=float("inf"),
                q25_log_cap=0.05,
                ratio_min=0.2,
                ratio_max=5.0,
                min_samples=10,
            )
        )
    )

    first, receipt = _station_bim_scale(records, cfg, samples_per_view=20)
    second, second_receipt = _station_bim_scale(records, cfg, samples_per_view=20)

    assert first == second
    assert receipt == second_receipt
    assert receipt["eligible_ratio_count"] == 288
    assert receipt["selected_ratio_count"] == 40
    assert first == pytest.approx(1.5)


def test_bootstrap_is_fixed_seed_and_resamples_room_clusters() -> None:
    rows = []
    for station, single, joint in (("a", 0.20, 0.15), ("b", 0.30, 0.25)):
        rows.extend(
            [
                {
                    "station_id": station,
                    "room": "shared_room",
                    "depth_method": "raw_da3",
                    "fusion_method": "single_best_view",
                    "abs_rel": single,
                },
                {
                    "station_id": station,
                    "room": "shared_room",
                    "depth_method": "raw_da3",
                    "fusion_method": "joint_huber",
                    "abs_rel": joint,
                },
            ]
        )
    kwargs = {
        "candidate_depth": "raw_da3",
        "candidate_fusion": "joint_huber",
        "reference_depth": "raw_da3",
        "reference_fusion": "single_best_view",
        "metric": "abs_rel",
        "seed": 7,
        "repetitions": 100,
    }

    first = _paired_bootstrap(rows, **kwargs)
    second = _paired_bootstrap(rows, **kwargs)

    assert first == second
    assert first["mean_difference"] == pytest.approx(-0.05)
    assert first["resampling_unit"] == "room cluster"
    assert first["room_count"] == 1
    assert first["station_count"] == 2
    assert first["candidate_better_room_fraction"] == 1.0


def test_room_cluster_bootstrap_retains_equal_station_estimand_with_unequal_rooms() -> None:
    rows = []
    for station, room, difference in (
        ("a", "large_room", 0.0),
        ("b", "large_room", 0.0),
        ("c", "small_room", 0.9),
    ):
        rows.extend(
            [
                {
                    "station_id": station,
                    "room": room,
                    "depth_method": "raw_da3",
                    "fusion_method": "candidate",
                    "spherical_abs_rel": 1.0 + difference,
                    "fixed_support_pixels": 10,
                },
                {
                    "station_id": station,
                    "room": room,
                    "depth_method": "raw_da3",
                    "fusion_method": "reference",
                    "spherical_abs_rel": 1.0,
                    "fixed_support_pixels": 10,
                },
            ]
        )

    standard = _paired_bootstrap(
        rows,
        candidate_depth="raw_da3",
        candidate_fusion="candidate",
        reference_depth="raw_da3",
        reference_fusion="reference",
        metric="spherical_abs_rel",
        seed=3,
        repetitions=200,
    )
    route = _route_paired_contrast(
        rows,
        kind="fixture",
        candidate={"fusion_method": "candidate"},
        reference={"fusion_method": "reference"},
        seed=3,
        repetitions=200,
    )

    assert standard["mean_difference"] == pytest.approx(0.3)
    assert route["primary_abs_rel_candidate"] - route["primary_abs_rel_reference"] == (
        pytest.approx(0.3)
    )
    assert route["room_cluster_paired_bootstrap_primary_abs_rel"][
        "mean_difference"
    ] == pytest.approx(0.3)


def test_cli_defaults_to_validation_and_requires_explicit_test_confirmation() -> None:
    args = parse_args([])
    assert args.split == "val"
    assert args.confirm_test is False
    assert args.tangent_manifest is None

    with pytest.raises(SystemExit):
        parse_args(["--split", "test"])

    test_args = parse_args(["--split", "test", "--confirm-test"])
    assert test_args.split == "test"
    assert test_args.confirm_test is True


def test_tangent_manifest_validation_projection_and_pano_only_route(tmp_path) -> None:
    fixture = _synthetic_tangent_manifest(tmp_path, pano_only=True)
    bundle = _validate_tangent_manifest(
        fixture.path,
        cfg=fixture.cfg,
        dataset=fixture.dataset,
        split="val",
        confirm_test=False,
        split_panoramas=[fixture.station],
    )
    args = parse_args(
        [
            "--pano-height",
            "8",
            "--sync-min-overlap",
            "1",
            "--sync-pair-max-samples",
            "128",
            "--bootstrap-repetitions",
            "10",
        ]
    )
    projected = _load_tangent_projected_views(
        bundle,
        fixture.station.camera_uuid,
        (8, 16),
        centrality_power=4.0,
    )
    prepared = _prepare_tangent_predictions(bundle, fixture.station, args)
    rows, _, info = _evaluate_tangent_only_station(
        fixture.station,
        prepared,
        fixture.cfg,
        args,
        pano_only=True,
    )

    assert len(projected) == 14
    assert all(np.all(view.base_weights > 0) for view in projected)
    assert tuple(prepared.fusions) == tuple(TANGENT_VARIANTS)
    assert len(rows) == 2 * len(TANGENT_VARIANTS) * len(TANGENT_FUSION_METHODS)
    assert {row["support_scope"] for row in rows} == {
        "native",
        "common_tangent6_tangent14",
    }
    assert all(row["pano_only"] is True for row in rows)
    assert info["pano_only"] is True
    common_counts = {
        row["fixed_support_pixels"]
        for row in rows
        if row["support_scope"] == "common_tangent6_tangent14"
    }
    assert len(common_counts) == 1


def test_tangent_manifest_tamper_and_test_authorization_fail_fast(tmp_path) -> None:
    fixture = _synthetic_tangent_manifest(tmp_path / "val", pano_only=True)
    tampered = json.loads(fixture.path.read_text(encoding="utf-8"))
    tampered["preset"]["views"][0]["intrinsic"][0][0] += 0.5
    fixture.path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="K/T/spec"):
        _validate_tangent_manifest(
            fixture.path,
            cfg=fixture.cfg,
            dataset=fixture.dataset,
            split="val",
            confirm_test=False,
            split_panoramas=[fixture.station],
        )

    test_fixture = _synthetic_tangent_manifest(tmp_path / "test", split="test")
    with pytest.raises(ValueError, match="runtime and cache authorization"):
        _validate_tangent_manifest(
            test_fixture.path,
            cfg=test_fixture.cfg,
            dataset=test_fixture.dataset,
            split="test",
            confirm_test=False,
            split_panoramas=[test_fixture.station],
        )


def test_tangent_manifest_resolves_relocated_content_by_sha(tmp_path) -> None:
    fixture = _synthetic_tangent_manifest(tmp_path, pano_only=True)
    payload = json.loads(fixture.path.read_text(encoding="utf-8"))
    payload["stations"][0]["pano_rgb"]["path"] = "/retired/machine/pano_rgb.png"
    for record in payload["stations"][0]["tangent_views"]:
        tangent_source = Path(record["tangent_rgb"]["path"])
        tangent_destination = (
            tmp_path / "tangent_rgb" / fixture.station.camera_uuid / tangent_source.name
        )
        tangent_destination.parent.mkdir(parents=True, exist_ok=True)
        tangent_source.rename(tangent_destination)
        record["tangent_rgb"]["path"] = f"/retired/machine/{tangent_source.name}"

        cache_source = Path(record["da3_cache"]["path"])
        cache_destination = tmp_path / "da3_cache" / cache_source.name
        cache_destination.parent.mkdir(parents=True, exist_ok=True)
        cache_source.rename(cache_destination)
        record["da3_cache"]["path"] = f"/retired/machine/{cache_source.name}"
    fixture.path.write_text(json.dumps(payload), encoding="utf-8")

    bundle = _validate_tangent_manifest(
        fixture.path,
        cfg=fixture.cfg,
        dataset=fixture.dataset,
        split="val",
        confirm_test=False,
        split_panoramas=[fixture.station],
    )

    assert all(path.is_file() for path in bundle.tangent_rgb_paths.values())
    assert all(path.is_file() for path in bundle.cache_paths.values())
    assert {path.parent.name for path in bundle.cache_paths.values()} == {"da3_cache"}


def test_zero_tangent_confidence_is_interpolated_as_zero_not_missing(tmp_path) -> None:
    fixture = _synthetic_tangent_manifest(tmp_path, pano_only=True)
    payload = json.loads(fixture.path.read_text(encoding="utf-8"))
    cache_record = payload["stations"][0]["tangent_views"][0]["da3_cache"]
    cache_path = Path(cache_record["path"])
    with np.load(cache_path, allow_pickle=False) as item:
        arrays = {key: item[key] for key in item.files}
    confidence = np.zeros((4, 4), dtype=np.float16)
    confidence[1:3, 1:3] = 1.0
    arrays["confidence"] = confidence
    np.savez_compressed(cache_path, **arrays)
    cache_record["sha256"] = _sha256(cache_path)
    fixture.path.write_text(json.dumps(payload), encoding="utf-8")
    bundle = _validate_tangent_manifest(
        fixture.path,
        cfg=fixture.cfg,
        dataset=fixture.dataset,
        split="val",
        confirm_test=False,
        split_panoramas=[fixture.station],
    )

    projected = _load_tangent_projected_views(
        bundle,
        fixture.station.camera_uuid,
        (8, 16),
        centrality_power=4.0,
    )[0]
    unit_range, _ = tangent_z_to_erp_range(
        np.ones((4, 4), dtype=np.float32),
        bundle.views[0],
        (8, 16),
    )
    centrality = 1.0 / unit_range.reshape(-1)[projected.indices]
    inferred_confidence = projected.base_weights / np.power(centrality, 4.0)

    assert np.any((inferred_confidence > 0.05) & (inferred_confidence < 0.95))
    assert float(inferred_confidence.max()) <= 1.0


def test_regular_pano_joint_uses_regular_support_for_quality_and_union_for_coverage() -> None:
    args = parse_args(
        [
            "--sync-min-overlap",
            "1",
            "--sync-pair-max-samples",
            "16",
            "--bootstrap-repetitions",
            "10",
        ]
    )
    regular_views = (_view("regular", [0, 1], [1.0, 2.0]),)
    tangent_views = tuple(
        _view(f"tangent/{index:02d}", [1, 2, 3], [2.0, 3.0, 4.0]) for index in range(14)
    )
    tangent = TangentPrepared(projected_views=tangent_views, fusions={})
    fusions = _prepare_regular_pano_joint(
        regular_views,
        tangent,
        (1, 4),
        args,
    )
    station = SimpleNamespace(camera_uuid="station", room="room")
    gt = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    rows, arrays, info = _evaluate_regular_pano_joint(
        station,
        fusions,
        gt,
        np.ones_like(gt, dtype=bool),
    )

    common = [row for row in rows if row["support_scope"] == "common_regular"]
    native = [row for row in rows if row["support_scope"] == "native_union"]
    assert {row["fixed_support_pixels"] for row in common} == {2}
    assert {
        row["fixed_support_pixels"] for row in native if row["source_set"] == "regular_only"
    } == {2}
    assert {
        row["fixed_support_pixels"]
        for row in native
        if row["source_set"] in {"regular_plus_tangent6", "regular_plus_tangent14"}
    } == {4}
    common_supports = [
        support for (scope, _, _), (_, _, support) in arrays.items() if scope == "common_regular"
    ]
    assert all(np.array_equal(support, common_supports[0]) for support in common_supports)
    assert np.array_equal(
        common_supports[0],
        np.asarray([[True, True, False, False]]),
    )
    gauges = {
        source: diagnostics["raw_da3"]["gauge"]
        for source, diagnostics in info["overlap_log_scale_synchronization"].items()
    }
    assert {tuple(gauge["anchor_frames"]) for gauge in gauges.values()} == {("regular",)}
    assert all(gauge["normalized_anchor_weights"] == [1.0] for gauge in gauges.values())
