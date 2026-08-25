from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.data.stanford_pano import (
    cv_rays_to_erp_pixels,
    discover_stanford_panoramas,
    erp_pixels_to_cv_rays,
    pano_range_image_to_regular_z,
    pano_range_to_regular_projection,
    pano_range_to_regular_z,
    regular_z_depth_to_pano,
    wrap_erp_horizontal,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_to_area(rotation: np.ndarray, location: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = location
    return transform


def _pose_payload(
    camera_uuid: str,
    room: str,
    frame_number: int | str,
    camera_to_area: np.ndarray,
) -> dict[str, object]:
    world_to_camera = np.linalg.inv(camera_to_area)
    return {
        "camera_k_matrix": [[800.0, 0.0, 540.0], [0.0, 805.0, 540.0], [0.0, 0.0, 1.0]],
        "camera_rt_matrix": world_to_camera[:3].tolist(),
        "camera_location": camera_to_area[:3, 3].tolist(),
        "camera_uuid": camera_uuid,
        "point_uuid": camera_uuid,
        "room": f"{room}_1",
        "frame_num": frame_number,
    }


def _write_area(
    root: Path,
    *,
    pano_centres: dict[str, np.ndarray],
    regular_centres: dict[str, list[np.ndarray]],
) -> Path:
    area_root = root / "area_1"
    for modality in ("rgb", "depth", "semantic", "pose"):
        (area_root / "pano" / modality).mkdir(parents=True, exist_ok=True)
    (area_root / "data" / "pose").mkdir(parents=True)

    for index, (camera_uuid, centre) in enumerate(pano_centres.items(), start=1):
        room = f"office_{index}"
        key = f"camera_{camera_uuid}_{room}_frame_equirectangular"
        for modality in ("rgb", "depth", "semantic"):
            (area_root / "pano" / modality / f"{key}_domain_{modality}.png").write_bytes(b"")
        pose = _pose_payload(
            camera_uuid,
            room,
            "equirectangular",
            _camera_to_area(_rotation_z(0.2 * index), centre),
        )
        (area_root / "pano" / "pose" / f"{key}_domain_pose.json").write_text(
            json.dumps(pose), encoding="utf-8"
        )

        for frame_number, regular_centre in enumerate(regular_centres.get(camera_uuid, [])):
            regular_key = f"camera_{camera_uuid}_{room}_frame_{frame_number}"
            regular_pose = _pose_payload(
                camera_uuid,
                room,
                frame_number,
                _camera_to_area(_rotation_z(-0.15 * (frame_number + 1)), regular_centre),
            )
            (area_root / "data" / "pose" / f"{regular_key}_domain_pose.json").write_text(
                json.dumps(regular_pose), encoding="utf-8"
            )

    # Discovery requires regular metadata to exist even when a particular pano is unpaired.
    if not any(regular_centres.values()):
        unrelated_uuid = "f" * 32
        unrelated_key = f"camera_{unrelated_uuid}_storage_1_frame_0"
        unrelated_pose = _pose_payload(
            unrelated_uuid,
            "storage_1",
            0,
            _camera_to_area(np.eye(3), np.zeros(3)),
        )
        (area_root / "data" / "pose" / f"{unrelated_key}_domain_pose.json").write_text(
            json.dumps(unrelated_pose), encoding="utf-8"
        )
    return area_root


def test_discovery_pairs_regular_metadata_and_allows_pano_only_station(tmp_path: Path) -> None:
    paired_uuid = "a" * 32
    pano_only_uuid = "b" * 32
    paired_centre = np.asarray([1.0, -2.0, 1.5])
    area_root = _write_area(
        tmp_path,
        pano_centres={paired_uuid: paired_centre, pano_only_uuid: np.zeros(3)},
        regular_centres={paired_uuid: [paired_centre, paired_centre], pano_only_uuid: []},
    )

    panoramas = discover_stanford_panoramas(area_root)

    assert len(panoramas) == 2
    by_uuid = {value.camera_uuid: value for value in panoramas}
    assert [view.frame_number for view in by_uuid[paired_uuid].regular_views] == [0, 1]
    assert by_uuid[pano_only_uuid].regular_views == ()
    assert by_uuid[paired_uuid].sample_id == f"office_1/{paired_uuid}/equirectangular"
    np.testing.assert_allclose(
        by_uuid[paired_uuid].world_to_camera @ by_uuid[paired_uuid].camera_to_area,
        np.eye(4),
        atol=1e-12,
    )


def test_discovery_rejects_regular_pano_centre_mismatch(tmp_path: Path) -> None:
    camera_uuid = "c" * 32
    area_root = _write_area(
        tmp_path,
        pano_centres={camera_uuid: np.zeros(3)},
        regular_centres={camera_uuid: [np.asarray([0.01, 0.0, 0.0])]},
    )
    with pytest.raises(ValueError, match="camera centres disagree"):
        discover_stanford_panoramas(area_root, center_tolerance_m=0.002)


def test_discovery_rejects_non_official_pose_shape(tmp_path: Path) -> None:
    camera_uuid = "d" * 32
    area_root = _write_area(
        tmp_path,
        pano_centres={camera_uuid: np.zeros(3)},
        regular_centres={camera_uuid: [np.zeros(3)]},
    )
    pano_pose_path = next((area_root / "pano" / "pose").glob("*.json"))
    payload = json.loads(pano_pose_path.read_text(encoding="utf-8"))
    payload["camera_rt_matrix"] = np.zeros((4, 3)).tolist()
    pano_pose_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="released 3x4"):
        discover_stanford_panoramas(area_root)


def test_erp_known_cv_axes_and_horizontal_wrap() -> None:
    shape = (200, 400)
    pixels = np.asarray(
        [
            [199.5, 99.5],  # front
            [299.5, 99.5],  # right
            [99.5, 99.5],  # left
            [199.5, -0.5],  # up
            [199.5, 199.5],  # down
        ]
    )
    expected = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    np.testing.assert_allclose(erp_pixels_to_cv_rays(pixels, shape), expected, atol=1e-15)
    np.testing.assert_allclose(
        erp_pixels_to_cv_rays(pixels + [shape[1] * 3, 0], shape), expected, atol=1e-15
    )
    np.testing.assert_allclose(wrap_erp_horizontal([-0.25, 0.0, 400.25], 400), [399.75, 0.0, 0.25])


def test_erp_pixel_ray_round_trip_is_subnanopixel() -> None:
    shape = (2048, 4096)
    pixels = np.asarray(
        [
            [0.0, 0.0],
            [4095.0, 2047.0],
            [2047.5, 1023.5],
            [-0.25, 300.125],
            [8197.75, 1700.875],
        ],
        dtype=np.float64,
    )
    reconstructed = cv_rays_to_erp_pixels(erp_pixels_to_cv_rays(pixels, shape), shape)
    expected_u = wrap_erp_horizontal(pixels[:, 0], shape[1])
    horizontal_error = np.mod(reconstructed[:, 0] - expected_u + shape[1] / 2, shape[1])
    horizontal_error -= shape[1] / 2
    np.testing.assert_allclose(horizontal_error, 0.0, atol=2e-10)
    np.testing.assert_allclose(reconstructed[:, 1], pixels[:, 1], atol=2e-10)


def test_regular_z_pano_range_projection_round_trip() -> None:
    pano_shape = (2048, 4096)
    intrinsic = np.asarray(
        [[910.0, 1.25, 540.0], [0.0, 905.0, 535.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    centre = np.asarray([2.0, -1.0, 1.6])
    regular_camera_to_area = _camera_to_area(_rotation_z(0.47), centre)
    pano_camera_to_area = _camera_to_area(_rotation_z(-1.13), centre)
    regular_world_to_camera = np.linalg.inv(regular_camera_to_area)
    pano_world_to_camera = np.linalg.inv(pano_camera_to_area)
    regular_pixels = np.asarray(
        [[0.0, 0.0], [1079.0, 1079.0], [540.25, 535.75], [215.125, 800.5]],
        dtype=np.float64,
    )
    z_depth = np.asarray([0.35, 4.8, 2.25, 1.1], dtype=np.float64)

    pano_pixels, radial_range = regular_z_depth_to_pano(
        regular_pixels,
        z_depth,
        intrinsic,
        regular_camera_to_area,
        pano_world_to_camera,
        pano_shape,
    )
    reconstructed_pixels, reconstructed_z, front_facing = pano_range_to_regular_projection(
        pano_pixels,
        radial_range,
        pano_shape,
        pano_camera_to_area,
        regular_world_to_camera,
        intrinsic,
    )

    assert front_facing.all()
    np.testing.assert_allclose(reconstructed_pixels, regular_pixels, atol=2e-10)
    np.testing.assert_allclose(reconstructed_z, z_depth, atol=2e-12)
    np.testing.assert_allclose(
        pano_range_to_regular_z(
            pano_pixels,
            radial_range,
            pano_shape,
            pano_camera_to_area,
            regular_world_to_camera,
        ),
        z_depth,
        atol=2e-12,
    )


def test_pano_range_image_reprojects_to_regular_z_across_erp_seam() -> None:
    pano_shape = (256, 512)
    regular_shape = (9, 11)
    intrinsic = np.asarray(
        [[8.0, 0.0, 5.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # A 180-degree yaw makes the regular view straddle the ERP seam.
    regular_camera_to_area = np.eye(4, dtype=np.float64)
    regular_camera_to_area[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    constant_range = np.full(pano_shape, 3.0, dtype=np.float32)

    depth, valid = pano_range_image_to_regular_z(
        constant_range,
        np.ones(pano_shape, dtype=bool),
        np.eye(4, dtype=np.float64),
        np.linalg.inv(regular_camera_to_area),
        intrinsic,
        regular_shape,
    )

    x, y = np.meshgrid(np.arange(regular_shape[1]), np.arange(regular_shape[0]))
    homogeneous = np.stack((x, y, np.ones(regular_shape)), axis=-1)
    rays = homogeneous @ np.linalg.inv(intrinsic).T
    expected = 3.0 * rays[..., 2] / np.linalg.norm(rays, axis=-1)
    assert valid.all()
    np.testing.assert_allclose(depth, expected, atol=2e-6)


def test_pano_range_image_normalizes_invalid_bilinear_neighbours() -> None:
    pano_shape = (64, 128)
    regular_shape = (3, 3)
    intrinsic = np.asarray(
        [[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    values = np.full(pano_shape, 2.0, dtype=np.float32)
    valid_input = np.ones(pano_shape, dtype=bool)
    # The front direction maps to the ERP centre.  Removing one neighbour must
    # not bias the normalized interpolation toward zero.
    valid_input[pano_shape[0] // 2, pano_shape[1] // 2] = False

    depth, valid = pano_range_image_to_regular_z(
        values,
        valid_input,
        np.eye(4),
        np.eye(4),
        intrinsic,
        regular_shape,
    )

    assert valid.all()
    assert np.all(depth > 0.0)
    expected_centre = 2.0
    assert depth[1, 1] == pytest.approx(expected_centre, abs=2e-6)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: erp_pixels_to_cv_rays(np.asarray([[0.0, -0.6]]), (10, 20)), "vertical"),
        (lambda: cv_rays_to_erp_pixels(np.asarray([[0.0, 0.0, 2.0]]), (10, 20)), "unit"),
        (lambda: cv_rays_to_erp_pixels(np.zeros((1, 3)), (10, 20)), "non-zero"),
        (lambda: wrap_erp_horizontal(np.asarray([np.nan]), 20), "non-finite"),
        (
            lambda: regular_z_depth_to_pano(
                np.asarray([[1.0, 2.0]]),
                np.asarray([-1.0]),
                np.eye(3),
                np.eye(4),
                np.eye(4),
                (10, 20),
            ),
            "positive",
        ),
        (
            lambda: regular_z_depth_to_pano(
                np.asarray([[1.0, 2.0]]),
                np.asarray([1.0]),
                np.eye(3),
                np.diag([2.0, 1.0, 1.0, 1.0]),
                np.eye(4),
                (10, 20),
            ),
            "orthonormal",
        ),
    ],
)
def test_geometry_fails_fast_on_invalid_inputs(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
