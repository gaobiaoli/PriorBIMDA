from __future__ import annotations

import numpy as np
import pytest

from bim_priorda3.data.pano_tangent import (
    PANO_TANGENT_PRESET_FOV_DEGREES,
    TangentViewSpec,
    build_cubemap_views,
    build_pano_tangent_preset,
    build_tangent_views,
    erp_range_to_tangent_z,
    erp_rgb_to_tangent,
    tangent_pixels_to_pano_rays,
    tangent_z_to_erp_range,
)
from bim_priorda3.data.stanford_pano import erp_pixels_to_cv_rays


def _centre_pixel(view_shape: tuple[int, int]) -> np.ndarray:
    height, width = view_shape
    return np.asarray([[(width - 1.0) / 2.0, (height - 1.0) / 2.0]])


def test_cubemap_has_canonical_directions_calibration_and_transforms() -> None:
    views = build_cubemap_views(5)
    assert [view.name for view in views] == ["front", "right", "back", "left", "up", "down"]
    expected_directions = {
        "front": [0.0, 0.0, 1.0],
        "right": [1.0, 0.0, 0.0],
        "back": [0.0, 0.0, -1.0],
        "left": [-1.0, 0.0, 0.0],
        "up": [0.0, -1.0, 0.0],
        "down": [0.0, 1.0, 0.0],
    }

    for view in views:
        np.testing.assert_allclose(
            tangent_pixels_to_pano_rays(_centre_pixel(view.image_shape), view)[0],
            expected_directions[view.name],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            view.intrinsic,
            [[2.5, 0.0, 2.0], [0.0, 2.5, 2.0], [0.0, 0.0, 1.0]],
            atol=1e-15,
        )
        np.testing.assert_allclose(view.T_face_from_pano[:3, 3], 0.0, atol=0.0)
        forward_in_face = view.T_face_from_pano[:3, :3] @ np.asarray(expected_directions[view.name])
        np.testing.assert_allclose(forward_in_face, [0.0, 0.0, 1.0], atol=1e-15)
        assert not view.intrinsic.flags.writeable
        assert not view.T_face_from_pano.flags.writeable

    front = views[0]
    boundary_rays = tangent_pixels_to_pano_rays(np.asarray([[-0.5, 2.0], [4.5, 2.0]]), front)
    np.testing.assert_allclose(
        boundary_rays,
        [[-np.sqrt(0.5), 0.0, np.sqrt(0.5)], [np.sqrt(0.5), 0.0, np.sqrt(0.5)]],
        atol=1e-15,
    )


def test_arbitrary_overlapping_tangent_specs_are_ordered_and_deterministic() -> None:
    specs = (
        TangentViewSpec("yaw_000", 0.0, 10.0, (48, 64), 100.0),
        TangentViewSpec("yaw_060", 60.0, 10.0, (48, 64), 100.0),
    )
    first = build_tangent_views(specs)
    second = build_tangent_views(specs)

    assert [view.name for view in first] == ["yaw_000", "yaw_060"]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.intrinsic, right.intrinsic)
        np.testing.assert_array_equal(left.T_face_from_pano, right.T_face_from_pano)
    first_forward = tangent_pixels_to_pano_rays(_centre_pixel((48, 64)), first[0])[0]
    np.testing.assert_allclose(
        first_forward,
        [0.0, -np.sin(np.deg2rad(10.0)), np.cos(np.deg2rad(10.0))],
        atol=1e-15,
    )


def test_nested14_is_a_strict_extension_of_overlapping_cubemap6() -> None:
    cubemap = build_pano_tangent_preset("cubemap6", 40)
    nested = build_pano_tangent_preset("nested14", 40)

    assert len(cubemap) == 6
    assert len(nested) == 14
    assert [view.name for view in nested[:6]] == [view.name for view in cubemap]
    for base, extended in zip(cubemap, nested[:6]):
        np.testing.assert_array_equal(base.intrinsic, extended.intrinsic)
        np.testing.assert_array_equal(base.T_face_from_pano, extended.T_face_from_pano)
        assert base.spec.horizontal_fov_degrees == PANO_TANGENT_PRESET_FOV_DEGREES
    corner = nested[6]
    direction = tangent_pixels_to_pano_rays(_centre_pixel(corner.image_shape), corner)[0]
    np.testing.assert_allclose(direction, np.asarray([1.0, -1.0, 1.0]) / np.sqrt(3.0), atol=1e-15)


def test_erp_rgb_sampling_uses_pixel_centres_and_wraps_across_seam() -> None:
    height, width = 4, 8
    erp = np.zeros((height, width, 3), dtype=np.float32)
    erp[..., 0] = np.arange(width, dtype=np.float32)[None, :]
    erp[..., 1] = np.arange(height, dtype=np.float32)[:, None]
    erp[..., 2] = 10.0
    by_name = {view.name: view for view in build_cubemap_views(1)}

    expected = {
        "front": [3.5, 1.5, 10.0],
        "right": [5.5, 1.5, 10.0],
        "left": [1.5, 1.5, 10.0],
        # Back points to ERP coordinate 7.5, midway between columns 7 and 0.
        "back": [3.5, 1.5, 10.0],
        # At either pole, vertical bilinear sampling clamps to the outer row.
        "up": [3.5, 0.0, 10.0],
        "down": [3.5, 3.0, 10.0],
    }
    for name, target in expected.items():
        rendered = erp_rgb_to_tangent(erp, by_name[name])
        assert rendered.dtype == erp.dtype
        np.testing.assert_allclose(rendered[0, 0], target, atol=1e-6)

    integer_rendered = erp_rgb_to_tangent((erp * 10.0).astype(np.uint8), by_name["front"])
    assert integer_rendered.dtype == np.uint8
    np.testing.assert_array_equal(integer_rendered[0, 0], [35, 15, 100])


def test_constant_erp_range_round_trip_through_tangent_z() -> None:
    radius = 3.25
    erp_shape = (64, 128)
    erp_range = np.full(erp_shape, radius, dtype=np.float32)
    front = build_cubemap_views(65)[0]

    z_depth, tangent_valid = erp_range_to_tangent_z(erp_range, front)
    assert tangent_valid.all()
    face_pixels = np.stack(
        np.meshgrid(np.arange(65, dtype=np.float64), np.arange(65, dtype=np.float64)), axis=-1
    )
    pano_rays = tangent_pixels_to_pano_rays(face_pixels, front)
    face_rays = pano_rays @ front.T_face_from_pano[:3, :3].T
    np.testing.assert_allclose(z_depth, radius * face_rays[..., 2], rtol=2e-7, atol=2e-7)

    reconstructed_range, erp_valid = tangent_z_to_erp_range(
        z_depth, front, erp_shape, valid_mask=tangent_valid, chunk_rows=7
    )
    assert erp_valid.any()
    # Bilinear interpolation of the smooth perspective-z field is the only
    # approximation in this closed loop. The largest error occurs in the
    # half-pixel sensor-boundary band, where sampling clamps to the outer pixel.
    np.testing.assert_allclose(reconstructed_range[erp_valid], radius, rtol=3e-3, atol=3e-4)


def test_invalid_depth_sample_does_not_poison_valid_bilinear_neighbours() -> None:
    erp_range = np.full((4, 8), 2.0, dtype=np.float32)
    erp_range[1, 3] = np.nan

    z_depth, valid = erp_range_to_tangent_z(erp_range, build_cubemap_views(1)[0])

    assert valid[0, 0]
    assert z_depth[0, 0] == pytest.approx(2.0)


def test_six_z_depth_contributions_cover_erp_and_convert_z_to_range() -> None:
    erp_shape = (32, 64)
    all_valid = np.zeros(erp_shape, dtype=np.bool_)
    erp_pixels = np.stack(
        np.meshgrid(
            np.arange(erp_shape[1], dtype=np.float64),
            np.arange(erp_shape[0], dtype=np.float64),
        ),
        axis=-1,
    )
    pano_rays = erp_pixels_to_cv_rays(erp_pixels, erp_shape)

    for view in build_cubemap_views(32):
        range_contribution, valid = tangent_z_to_erp_range(
            np.full(view.image_shape, 2.0, dtype=np.float32),
            view,
            erp_shape,
            chunk_rows=5,
        )
        face_rays = pano_rays @ view.T_face_from_pano[:3, :3].T
        np.testing.assert_allclose(
            range_contribution[valid] * face_rays[..., 2][valid],
            2.0,
            rtol=1e-7,
            atol=1e-7,
        )
        assert np.all(range_contribution[~valid] == 0.0)
        all_valid |= valid

    assert all_valid.all()


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: TangentViewSpec("bad", 0.0, 91.0, (8, 8)),
            ValueError,
            "pitch_degrees",
        ),
        (
            lambda: TangentViewSpec("bad", 0.0, 0.0, (8, 8), 180.0),
            ValueError,
            "strictly",
        ),
        (lambda: build_cubemap_views(True), TypeError, "positive integer"),
        (lambda: build_tangent_views(()), ValueError, "at least one"),
        (
            lambda: build_tangent_views(
                (
                    TangentViewSpec("same", 0.0, 0.0, (8, 8)),
                    TangentViewSpec("same", 20.0, 0.0, (8, 8)),
                )
            ),
            ValueError,
            "unique",
        ),
        (
            lambda: erp_rgb_to_tangent(
                np.zeros((8, 16), dtype=np.uint8), build_cubemap_views(8)[0]
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: tangent_pixels_to_pano_rays(
                np.asarray([[8.0, 0.0]]), build_cubemap_views(8)[0]
            ),
            ValueError,
            "outside",
        ),
        (
            lambda: tangent_z_to_erp_range(np.ones((7, 8)), build_cubemap_views(8)[0], (8, 16)),
            ValueError,
            "does not match",
        ),
        (
            lambda: erp_range_to_tangent_z(
                np.ones((8, 16)),
                build_cubemap_views(8)[0],
                valid_mask=np.ones((8, 16), dtype=np.uint8),
            ),
            TypeError,
            "boolean",
        ),
        (
            lambda: tangent_z_to_erp_range(
                np.ones((8, 8)), build_cubemap_views(8)[0], (8, 16), chunk_rows=0
            ),
            ValueError,
            "positive integer",
        ),
    ],
)
def test_pano_tangent_fails_fast(call: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        call()  # type: ignore[operator]
