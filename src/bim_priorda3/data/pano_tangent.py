"""Model-independent equirectangular panorama/tangent-view geometry.

The panorama camera and every tangent camera use OpenCV axes: ``+x`` right,
``+y`` down, and ``+z`` forward.  Array indices are pixel-centre coordinates;
therefore the image sensor spans ``[-0.5, width - 0.5]`` horizontally and
``[-0.5, height - 0.5]`` vertically.  ERP longitude wraps horizontally.

Perspective depth is explicitly ``z`` depth, while panorama depth is radial
range.  Keeping those quantities separate prevents a common, direction-
dependent scale error when merging predictions on the sphere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from bim_priorda3.data.stanford_pano import cv_rays_to_erp_pixels, erp_pixels_to_cv_rays

_EPS = 1e-12
PANO_TANGENT_PRESETS = ("cubemap6", "nested14")
PANO_TANGENT_PRESET_FOV_DEGREES = 100.0


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _image_shape(value: tuple[int, int], name: str = "image_shape") -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a (height, width) tuple, got {value!r}")
    return (
        _positive_integer(value[0], f"{name}[0]"),
        _positive_integer(value[1], f"{name}[1]"),
    )


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real number, got {value!r}") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TangentViewSpec:
    """Declarative definition of one pinhole view on an ERP panorama.

    ``yaw_degrees`` is positive to the panorama's right. ``pitch_degrees`` is
    positive upward, despite the OpenCV ``+y``-down camera axis. Positive roll
    rotates the view's right axis toward its down axis. The horizontal field of
    view is measured at the sensor boundaries, not at the outer pixel centres.
    Pixels are square; the vertical field of view follows from the aspect ratio.
    """

    name: str
    yaw_degrees: float
    pitch_degrees: float
    image_shape: tuple[int, int]
    horizontal_fov_degrees: float = 90.0
    roll_degrees: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tangent view name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("Tangent view name must not have leading or trailing whitespace")
        yaw = _finite_float(self.yaw_degrees, "yaw_degrees")
        pitch = _finite_float(self.pitch_degrees, "pitch_degrees")
        roll = _finite_float(self.roll_degrees, "roll_degrees")
        horizontal_fov = _finite_float(self.horizontal_fov_degrees, "horizontal_fov_degrees")
        if pitch < -90.0 or pitch > 90.0:
            raise ValueError(f"pitch_degrees must lie in [-90, 90], got {pitch}")
        if horizontal_fov <= 0.0 or horizontal_fov >= 180.0:
            raise ValueError(
                f"horizontal_fov_degrees must lie strictly between 0 and 180, got {horizontal_fov}"
            )
        object.__setattr__(self, "yaw_degrees", yaw)
        object.__setattr__(self, "pitch_degrees", pitch)
        object.__setattr__(self, "roll_degrees", roll)
        object.__setattr__(self, "horizontal_fov_degrees", horizontal_fov)
        object.__setattr__(self, "image_shape", _image_shape(self.image_shape))


@dataclass(frozen=True)
class TangentView:
    """Materialized tangent camera, including its calibration and pose.

    ``T_face_from_pano`` maps homogeneous points from the panorama camera into
    the face camera. Both cameras have the same optical centre, so translation
    is exactly zero.
    """

    spec: TangentViewSpec
    intrinsic: np.ndarray
    T_face_from_pano: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TangentViewSpec):
            raise TypeError("spec must be a TangentViewSpec")
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("intrinsic must be a finite 3x3 matrix")
        if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
            raise ValueError("intrinsic must have homogeneous row [0, 0, 1]")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("intrinsic must have positive focal lengths")

        transform = np.asarray(self.T_face_from_pano, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("T_face_from_pano must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
            raise ValueError("T_face_from_pano must be homogeneous")
        if not np.allclose(transform[:3, 3], 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("A tangent view must share the panorama optical centre")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10, rtol=0.0):
            raise ValueError("T_face_from_pano rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("T_face_from_pano rotation must be proper")
        object.__setattr__(self, "intrinsic", _readonly_float64(intrinsic))
        object.__setattr__(self, "T_face_from_pano", _readonly_float64(transform))

    @property
    def name(self) -> str:
        """Stable name of this tangent view."""

        return self.spec.name

    @property
    def image_shape(self) -> tuple[int, int]:
        """``(height, width)`` of this tangent view."""

        return self.spec.image_shape


def build_tangent_view(spec: TangentViewSpec) -> TangentView:
    """Materialize a deterministic pinhole camera from a tangent-view spec."""

    if not isinstance(spec, TangentViewSpec):
        raise TypeError("spec must be a TangentViewSpec")
    height, width = spec.image_shape
    half_fov = np.deg2rad(spec.horizontal_fov_degrees) / 2.0
    focal = float(width) / (2.0 * np.tan(half_fov))
    intrinsic = np.asarray(
        [
            [focal, 0.0, (float(width) - 1.0) / 2.0],
            [0.0, focal, (float(height) - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    yaw = np.deg2rad(spec.yaw_degrees)
    pitch = np.deg2rad(spec.pitch_degrees)
    roll = np.deg2rad(spec.roll_degrees)
    forward = np.asarray([np.cos(pitch) * np.sin(yaw), -np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    right = np.asarray([np.cos(yaw), 0.0, -np.sin(yaw)])
    down = np.cross(forward, right)
    rolled_right = np.cos(roll) * right + np.sin(roll) * down
    rolled_down = -np.sin(roll) * right + np.cos(roll) * down
    rotation_pano_from_face = np.stack((rolled_right, rolled_down, forward), axis=1)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_pano_from_face.T
    return TangentView(spec=spec, intrinsic=intrinsic, T_face_from_pano=transform)


def build_tangent_views(specs: Sequence[TangentViewSpec]) -> tuple[TangentView, ...]:
    """Materialize an ordered set of arbitrary, possibly overlapping views."""

    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence of TangentViewSpec values")
    views = tuple(build_tangent_view(spec) for spec in specs)
    if not views:
        raise ValueError("specs must contain at least one tangent view")
    names = [view.name for view in views]
    if len(names) != len(set(names)):
        raise ValueError("Tangent view names must be unique")
    return views


def cubemap_view_specs(face_resolution: int) -> tuple[TangentViewSpec, ...]:
    """Return canonical front/right/back/left/up/down 90-degree face specs."""

    size = _positive_integer(face_resolution, "face_resolution")
    shape = (size, size)
    return (
        TangentViewSpec("front", 0.0, 0.0, shape),
        TangentViewSpec("right", 90.0, 0.0, shape),
        TangentViewSpec("back", 180.0, 0.0, shape),
        TangentViewSpec("left", -90.0, 0.0, shape),
        TangentViewSpec("up", 0.0, 90.0, shape),
        TangentViewSpec("down", 0.0, -90.0, shape),
    )


def build_cubemap_views(face_resolution: int) -> tuple[TangentView, ...]:
    """Build the canonical six-face, 90-degree-FOV cubemap."""

    return build_tangent_views(cubemap_view_specs(face_resolution))


def pano_tangent_preset_specs(
    preset: str,
    face_resolution: int,
) -> tuple[TangentViewSpec, ...]:
    """Return a named, overlapping panorama-inference preset.

    ``cubemap6`` uses the six canonical cube directions with a 100-degree FOV,
    giving overlap while retaining complete spherical coverage. ``nested14``
    appends the eight cube-corner directions. Its first six entries are exactly
    ``cubemap6``, which makes comparisons nested rather than confounded by a
    changed base projection.
    """

    if not isinstance(preset, str):
        raise TypeError("preset must be a string")
    if preset not in PANO_TANGENT_PRESETS:
        raise ValueError(
            f"Unknown pano tangent preset {preset!r}; expected one of {PANO_TANGENT_PRESETS}"
        )
    size = _positive_integer(face_resolution, "face_resolution")
    shape = (size, size)
    field_of_view = PANO_TANGENT_PRESET_FOV_DEGREES
    base = (
        TangentViewSpec("front", 0.0, 0.0, shape, field_of_view),
        TangentViewSpec("right", 90.0, 0.0, shape, field_of_view),
        TangentViewSpec("back", 180.0, 0.0, shape, field_of_view),
        TangentViewSpec("left", -90.0, 0.0, shape, field_of_view),
        TangentViewSpec("up", 0.0, 90.0, shape, field_of_view),
        TangentViewSpec("down", 0.0, -90.0, shape, field_of_view),
    )
    if preset == "cubemap6":
        return base

    corner_pitch = float(np.rad2deg(np.arcsin(1.0 / np.sqrt(3.0))))
    corners = (
        TangentViewSpec("corner_right_up_front", 45.0, corner_pitch, shape, field_of_view),
        TangentViewSpec("corner_right_up_back", 135.0, corner_pitch, shape, field_of_view),
        TangentViewSpec("corner_left_up_back", -135.0, corner_pitch, shape, field_of_view),
        TangentViewSpec("corner_left_up_front", -45.0, corner_pitch, shape, field_of_view),
        TangentViewSpec("corner_right_down_front", 45.0, -corner_pitch, shape, field_of_view),
        TangentViewSpec("corner_right_down_back", 135.0, -corner_pitch, shape, field_of_view),
        TangentViewSpec("corner_left_down_back", -135.0, -corner_pitch, shape, field_of_view),
        TangentViewSpec("corner_left_down_front", -45.0, -corner_pitch, shape, field_of_view),
    )
    return base + corners


def build_pano_tangent_preset(preset: str, face_resolution: int) -> tuple[TangentView, ...]:
    """Materialize a named panorama-inference preset."""

    return build_tangent_views(pano_tangent_preset_specs(preset, face_resolution))


def _pixel_grid(
    image_shape: tuple[int, int], row_start: int = 0, row_stop: int | None = None
) -> np.ndarray:
    height, width = _image_shape(image_shape)
    stop = height if row_stop is None else row_stop
    if row_start < 0 or stop < row_start or stop > height:
        raise ValueError(f"Invalid row interval [{row_start}, {stop}) for height {height}")
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(row_start, stop, dtype=np.float64),
    )
    return np.stack((u, v), axis=-1)


def tangent_pixels_to_pano_rays(pixels: np.ndarray, view: TangentView) -> np.ndarray:
    """Convert tangent pixel-centre coordinates to panorama-camera unit rays."""

    if not isinstance(view, TangentView):
        raise TypeError("view must be a TangentView")
    values = np.asarray(pixels, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError(f"pixels must have shape (..., 2), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("pixels contain non-finite coordinates")
    height, width = view.image_shape
    if (
        np.any(values[..., 0] < -0.5)
        or np.any(values[..., 0] > width - 0.5)
        or np.any(values[..., 1] < -0.5)
        or np.any(values[..., 1] > height - 0.5)
    ):
        raise ValueError("pixels lie outside the tangent image sensor boundaries")

    homogeneous = np.concatenate(
        (values, np.ones(values.shape[:-1] + (1,), dtype=np.float64)), axis=-1
    )
    face_rays = homogeneous @ np.linalg.inv(view.intrinsic).T
    face_rays /= np.linalg.norm(face_rays, axis=-1, keepdims=True)
    rotation_pano_from_face = view.T_face_from_pano[:3, :3].T
    pano_rays = face_rays @ rotation_pano_from_face.T
    pano_rays /= np.linalg.norm(pano_rays, axis=-1, keepdims=True)
    return pano_rays


def _validate_erp_rgb(erp_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(erp_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"erp_rgb must have shape (height, width, 3), got {image.shape}")
    _image_shape((image.shape[0], image.shape[1]), "erp_rgb shape")
    if image.dtype.kind not in "uif":
        raise TypeError(f"erp_rgb must have an integer or floating dtype, got {image.dtype}")
    if image.dtype.kind == "f" and not np.isfinite(image).all():
        raise ValueError("erp_rgb contains non-finite values")
    return image


def _validate_range_image(
    values: np.ndarray,
    valid_mask: np.ndarray | None,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(values)
    if image.ndim != 2:
        raise ValueError(f"{name} must have shape (height, width), got {image.shape}")
    _image_shape((image.shape[0], image.shape[1]), f"{name} shape")
    if image.dtype.kind not in "uif":
        raise TypeError(f"{name} must have an integer or floating dtype, got {image.dtype}")
    numeric = image.astype(np.float64, copy=False)
    valid = np.isfinite(numeric) & (numeric > 0.0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask)
        if supplied.dtype != np.bool_:
            raise TypeError("valid_mask must have boolean dtype")
        if supplied.shape != image.shape:
            raise ValueError(
                f"valid_mask shape {supplied.shape} does not match {name} shape {image.shape}"
            )
        valid &= supplied
    return numeric, valid


def _bilinear_sample(
    image: np.ndarray,
    pixels: np.ndarray,
    *,
    horizontal_wrap: bool,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample scalar or channel-last data at pixel-centre coordinates."""

    height, width = image.shape[:2]
    u = pixels[..., 0]
    v = pixels[..., 1]
    x0_raw = np.floor(u).astype(np.int64)
    y0_raw = np.floor(v).astype(np.int64)
    x1_raw = x0_raw + 1
    y1_raw = y0_raw + 1
    wx = u - x0_raw
    wy = v - y0_raw
    if horizontal_wrap:
        x0 = np.mod(x0_raw, width)
        x1 = np.mod(x1_raw, width)
    else:
        x0 = np.clip(x0_raw, 0, width - 1)
        x1 = np.clip(x1_raw, 0, width - 1)
    y0 = np.clip(y0_raw, 0, height - 1)
    y1 = np.clip(y1_raw, 0, height - 1)
    weights = (
        (1.0 - wx) * (1.0 - wy),
        wx * (1.0 - wy),
        (1.0 - wx) * wy,
        wx * wy,
    )
    indices = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    channels = image.ndim - 2
    result_shape = pixels.shape[:-1] + image.shape[2:]
    result = np.zeros(result_shape, dtype=np.float64)
    support = np.zeros(pixels.shape[:-1], dtype=np.float64)
    source_valid = (
        np.ones((height, width), dtype=np.bool_)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=np.bool_)
    )
    for weight, (row, column) in zip(weights, indices):
        sample_valid = source_valid[row, column]
        accepted_weight = weight * sample_valid
        expanded_weight = accepted_weight[(...,) + (None,) * channels]
        expanded_valid = sample_valid[(...,) + (None,) * channels]
        sample = np.where(expanded_valid, image[row, column], 0.0)
        result += expanded_weight * sample
        support += accepted_weight
    valid = support > _EPS
    divisor = support[(...,) + (None,) * channels]
    np.divide(result, divisor, out=result, where=divisor > _EPS)
    return result, valid


def _cast_interpolated_rgb(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype.kind in "ui":
        limits = np.iinfo(dtype)
        return np.clip(np.rint(values), limits.min, limits.max).astype(dtype)
    return values.astype(dtype, copy=False)


def erp_rgb_to_tangent(erp_rgb: np.ndarray, view: TangentView) -> np.ndarray:
    """Render an ERP RGB array into one tangent view with bilinear sampling.

    The input dtype is preserved. Integer RGB is rounded only after interpolation.
    Horizontal samples wrap across the ERP seam; vertical samples clamp at poles.
    """

    if not isinstance(view, TangentView):
        raise TypeError("view must be a TangentView")
    image = _validate_erp_rgb(erp_rgb)
    pixels = _pixel_grid(view.image_shape)
    pano_rays = tangent_pixels_to_pano_rays(pixels, view)
    erp_pixels = cv_rays_to_erp_pixels(pano_rays, image.shape[:2])
    sampled, valid = _bilinear_sample(image, erp_pixels, horizontal_wrap=True)
    if not valid.all():  # ERP covers the complete sphere, so this is an invariant.
        raise AssertionError("ERP RGB sampling unexpectedly produced invalid tangent pixels")
    return _cast_interpolated_rgb(sampled, image.dtype)


def erp_range_to_tangent_z(
    erp_range: np.ndarray,
    view: TangentView,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ERP radial range into tangent-camera ``z`` depth and a valid mask."""

    if not isinstance(view, TangentView):
        raise TypeError("view must be a TangentView")
    radial_range, source_valid = _validate_range_image(erp_range, valid_mask, "erp_range")
    pixels = _pixel_grid(view.image_shape)
    pano_rays = tangent_pixels_to_pano_rays(pixels, view)
    erp_pixels = cv_rays_to_erp_pixels(pano_rays, radial_range.shape)
    sampled_range, valid = _bilinear_sample(
        radial_range,
        erp_pixels,
        horizontal_wrap=True,
        valid_mask=source_valid,
    )
    face_rays = pano_rays @ view.T_face_from_pano[:3, :3].T
    z_depth = sampled_range * face_rays[..., 2]
    valid &= np.isfinite(z_depth) & (z_depth > _EPS)
    z_depth[~valid] = 0.0
    return z_depth.astype(np.float32), valid


def tangent_z_to_erp_range(
    z_depth: np.ndarray,
    view: TangentView,
    erp_shape: tuple[int, int],
    *,
    valid_mask: np.ndarray | None = None,
    chunk_rows: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject tangent ``z`` depth as one dense ERP range contribution.

    The returned range is zero outside the view or where no valid source depth
    contributes. The boolean mask distinguishes those zeros from measurements.
    Processing is row-chunked to keep 4096x2048 panorama memory bounded.
    """

    if not isinstance(view, TangentView):
        raise TypeError("view must be a TangentView")
    depth, source_valid = _validate_range_image(z_depth, valid_mask, "z_depth")
    if depth.shape != view.image_shape:
        raise ValueError(
            f"z_depth shape {depth.shape} does not match tangent view {view.image_shape}"
        )
    height, width = _image_shape(erp_shape, "erp_shape")
    rows_per_chunk = _positive_integer(chunk_rows, "chunk_rows")
    output = np.zeros((height, width), dtype=np.float32)
    output_valid = np.zeros((height, width), dtype=np.bool_)
    rotation_face_from_pano = view.T_face_from_pano[:3, :3]
    camera_matrix = view.intrinsic
    face_height, face_width = view.image_shape

    for row_start in range(0, height, rows_per_chunk):
        row_stop = min(row_start + rows_per_chunk, height)
        erp_pixels = _pixel_grid((height, width), row_start, row_stop)
        pano_rays = erp_pixels_to_cv_rays(erp_pixels, (height, width))
        face_rays = pano_rays @ rotation_face_from_pano.T
        face_z = face_rays[..., 2]
        front = face_z > _EPS
        face_pixels = np.zeros(erp_pixels.shape, dtype=np.float64)
        face_pixels[..., 0] = np.where(
            front,
            camera_matrix[0, 0] * face_rays[..., 0] / np.maximum(face_z, _EPS)
            + camera_matrix[0, 2],
            0.0,
        )
        face_pixels[..., 1] = np.where(
            front,
            camera_matrix[1, 1] * face_rays[..., 1] / np.maximum(face_z, _EPS)
            + camera_matrix[1, 2],
            0.0,
        )
        inside = (
            front
            & (face_pixels[..., 0] >= -0.5)
            & (face_pixels[..., 0] <= face_width - 0.5)
            & (face_pixels[..., 1] >= -0.5)
            & (face_pixels[..., 1] <= face_height - 0.5)
        )
        sampled_z, sampled_valid = _bilinear_sample(
            depth,
            face_pixels,
            horizontal_wrap=False,
            valid_mask=source_valid,
        )
        accepted = inside & sampled_valid & np.isfinite(sampled_z) & (sampled_z > 0.0)
        range_chunk = np.zeros(sampled_z.shape, dtype=np.float64)
        np.divide(sampled_z, face_z, out=range_chunk, where=accepted)
        accepted &= np.isfinite(range_chunk) & (range_chunk > 0.0)
        range_chunk[~accepted] = 0.0
        output[row_start:row_stop] = range_chunk.astype(np.float32)
        output_valid[row_start:row_stop] = accepted
    return output, output_valid


__all__ = [
    "PANO_TANGENT_PRESETS",
    "PANO_TANGENT_PRESET_FOV_DEGREES",
    "TangentView",
    "TangentViewSpec",
    "build_cubemap_views",
    "build_pano_tangent_preset",
    "build_tangent_view",
    "build_tangent_views",
    "cubemap_view_specs",
    "erp_range_to_tangent_z",
    "erp_rgb_to_tangent",
    "pano_tangent_preset_specs",
    "tangent_pixels_to_pano_rays",
    "tangent_z_to_erp_range",
]
