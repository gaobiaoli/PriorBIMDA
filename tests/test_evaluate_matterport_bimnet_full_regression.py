import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "model"
    / "evaluate_matterport_bimnet_full_regression.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_matterport_bimnet_full_regression",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_processed_geometry_matches_matterport_da3_protocol():
    intrinsics = np.array(
        [[1072.25, 0.0, 638.382], [0.0, 1072.15, 521.447], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    height, width, processed, focal_scale = MODULE.processed_geometry(
        1024,
        1280,
        intrinsics,
        504,
    )
    assert (height, width) == (406, 504)
    np.testing.assert_allclose(processed[0, 0], 422.1984375)
    np.testing.assert_allclose(processed[1, 1], 425.09072265625)
    np.testing.assert_allclose(focal_scale, 1.4121486002604168)


def test_predict_scale_reconstructs_pseudo_huber_round_updates_from_zero():
    class PseudoHuberModel:
        @staticmethod
        def _estimate_attention_scale(batch, base_depth):
            assert batch["base_depth"] is base_depth
            return {
                "iteration_log_scales": torch.tensor([[[0.10], [0.16], [0.14]]]),
                "log_scale": torch.tensor([[[[0.14]]]]),
                "scale": torch.exp(torch.tensor([[[[0.14]]]])),
                "pixel_support": torch.tensor([123]),
                "token_support": torch.tensor([45]),
            }

    batch = {"base_depth": torch.ones(1, 1, 2, 2)}
    prediction = MODULE.predict_scale(PseudoHuberModel(), batch)
    np.testing.assert_allclose(
        [prediction[f"round_{index}_update"] for index in range(1, 4)],
        [0.10, 0.06, -0.02],
        atol=1e-7,
    )
    assert prediction["pixel_support"] == 123
    assert prediction["token_support"] == 45


def test_aggregate_rows_reconstructs_micro_metrics_and_paired_improvement():
    rows = []
    for valid_pixels, raw_abs_rel, learned_abs_rel, raw_rmse, learned_rmse in (
        (1, 0.4, 0.2, 1.0, 0.5),
        (3, 0.2, 0.1, 3.0, 1.5),
    ):
        row = {
            "status": "ok",
            "gt_valid_pixels": valid_pixels,
            "learned_scale": 0.9,
            "scale_log_error": 0.1,
        }
        for prefix, abs_rel, rmse in (
            ("raw", raw_abs_rel, raw_rmse),
            ("learned", learned_abs_rel, learned_rmse),
            ("oracle_frame_scale", 0.05, 0.25),
        ):
            for metric in MODULE.METRICS:
                row[f"{prefix}_{metric}"] = abs_rel
            row[f"{prefix}_rmse_m"] = rmse
            row[f"{prefix}_rmse_log"] = rmse / 10
            row[f"{prefix}_mean_log_error"] = 0.0
        rows.append(row)

    result = MODULE.aggregate_rows(rows)
    raw = result["predictions"]["raw"]["pixel_micro"]
    learned = result["predictions"]["learned"]["pixel_micro"]
    np.testing.assert_allclose(raw["abs_rel"], 0.25)
    np.testing.assert_allclose(learned["abs_rel"], 0.125)
    np.testing.assert_allclose(raw["rmse_m"], np.sqrt(7.0))
    np.testing.assert_allclose(learned["rmse_m"], np.sqrt(1.75))
    assert result["learned_vs_raw"]["pixel_micro_abs_rel_relative_improvement"] == 0.5
    assert result["learned_vs_raw"]["frame_win_fraction"] == 1.0


def test_zero_gt_is_rejected_before_bim_render_or_model_inference(tmp_path):
    class MustNotRun:
        def __getattr__(self, name):
            raise AssertionError(f"{name} must not be used for an all-zero GT frame")

    frame = SimpleNamespace(
        scene_id="scene",
        panorama_id="pano",
        frame_id="pano_i0_0",
        camera_index=0,
        yaw_index=0,
        rgb_path=tmp_path / "rgb.jpg",
        depth_path=tmp_path / "depth.png",
        image_shape=(1024, 1280),
        intrinsics=np.array(
            [[1072.25, 0.0, 638.382], [0.0, 1072.15, 521.447], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        depth=np.zeros((1024, 1280), dtype=np.float32),
    )
    args = SimpleNamespace(process_res=504)
    row = MODULE.evaluate_frame(
        frame=frame,
        da3_model=MustNotRun(),
        scale_model=MustNotRun(),
        raycaster=MustNotRun(),
        args=args,
        model_min_support=100,
    )
    assert row["status"] == "skipped_bad_gt"
    assert row["gt_valid_pixels"] == 0
    assert row["filter_reasons"] == "gt_zero_depth"
    assert row["effective_pass"] is False
