#!/usr/bin/env python3
"""Visualize a BIMSyn room IFC and its relationship to one Stanford regular view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import ifcopenshell
import ifcopenshell.geom
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.patches import Patch

from bim_priorda3.data.ifc_envelope import ENVELOPE_CATEGORIES, envelope_category
from bim_priorda3.data.stanford2d3ds import pose_matrices

CLASS_NAMES = (*ENVELOPE_CATEGORIES, "furniture", "proxy", "other")
CLASS_COLORS = np.asarray(
    (
        (0.76, 0.80, 0.84),  # wall
        (0.60, 0.72, 0.56),  # floor
        (0.86, 0.82, 0.68),  # ceiling
        (0.57, 0.43, 0.30),  # door
        (0.42, 0.70, 0.84),  # window
        (0.53, 0.65, 0.76),  # column
        (0.36, 0.51, 0.66),  # beam
        (0.90, 0.48, 0.20),  # furniture
        (0.61, 0.39, 0.70),  # proxy
        (0.55, 0.55, 0.55),  # other
    ),
    dtype=np.float32,
)
BACKGROUND = np.asarray((0.965, 0.965, 0.95), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--ifc-dir", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=504)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def find_record(manifest: Path, sample_id: str) -> dict[str, object]:
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record["id"]) == sample_id:
                return record
    raise KeyError(f"Sample {sample_id!r} is absent from {manifest}")


def product_class_index(product: object) -> int:
    envelope = envelope_category(product)
    if envelope is not None:
        return ENVELOPE_CATEGORIES.index(envelope)
    if product.is_a("IfcFurnishingElement"):
        return CLASS_NAMES.index("furniture")
    if product.is_a("IfcBuildingElementProxy"):
        return CLASS_NAMES.index("proxy")
    return CLASS_NAMES.index("other")


def load_full_ifc(
    path: Path,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    model = ifcopenshell.open(str(path))
    products = [
        product
        for product in model.by_type("IfcProduct")
        if getattr(product, "Representation", None) is not None
    ]
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("convert-back-units", False)
    settings.set("weld-vertices", True)
    iterator = ifcopenshell.geom.iterator(
        settings,
        model,
        max(1, workers),
        include=products,
    )
    if not iterator.initialize():
        raise RuntimeError(f"IfcOpenShell could not initialize geometry for {path}")

    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    triangle_classes: list[np.ndarray] = []
    object_counts = {name: 0 for name in CLASS_NAMES}
    offset = 0
    while True:
        shape = iterator.get()
        product = model.by_guid(shape.guid)
        product_vertices = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        product_triangles = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        if len(product_vertices) and len(product_triangles):
            class_index = product_class_index(product)
            vertices.append(product_vertices.astype(np.float32))
            triangles.append((product_triangles + offset).astype(np.int32))
            triangle_classes.append(
                np.full(len(product_triangles), class_index, dtype=np.uint8)
            )
            object_counts[CLASS_NAMES[class_index]] += 1
            offset += len(product_vertices)
        if not iterator.next():
            break
    if not triangles:
        raise RuntimeError(f"No renderable IFC geometry found in {path}")
    return (
        np.concatenate(vertices),
        np.concatenate(triangles),
        np.concatenate(triangle_classes),
        {name: count for name, count in object_counts.items() if count},
    )


def make_scene(vertices: np.ndarray, triangles: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.ascontiguousarray(vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.ascontiguousarray(triangles), dtype=o3d.core.Dtype.Int32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def category_render(
    scene: o3d.t.geometry.RaycastingScene,
    face_classes: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    distance = result["t_hit"].numpy().astype(np.float32)
    primitive = result["primitive_ids"].numpy().astype(np.int64)
    valid = np.isfinite(distance) & (primitive >= 0) & (primitive < len(face_classes))
    image = np.broadcast_to(BACKGROUND, (*distance.shape, 3)).copy()
    normals = result["primitive_normals"].numpy().astype(np.float32)
    light = np.asarray((-0.35, -0.45, 0.82), dtype=np.float32)
    light /= np.linalg.norm(light)
    illumination = 0.58 + 0.42 * np.abs(np.sum(normals * light, axis=-1))
    image[valid] = CLASS_COLORS[face_classes[primitive[valid]]] * illumination[valid, None]

    valid_u8 = (valid.astype(np.uint8) * 255)
    outline = cv2.Canny(valid_u8, 20, 80) > 0
    image[outline] = 0.12
    return np.clip(image, 0.0, 1.0), distance


def camera_rays(
    intrinsic: np.ndarray,
    camera_to_area: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    directions_camera = pixels @ np.linalg.inv(intrinsic).T
    directions_area = directions_camera @ camera_to_area[:3, :3].T
    origins = np.broadcast_to(camera_to_area[:3, 3], directions_area.shape)
    return origins.astype(np.float32), directions_area.astype(np.float32)


def overview_rays(
    vertices: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    lower = vertices.min(axis=0).astype(np.float64)
    upper = vertices.max(axis=0).astype(np.float64)
    center = 0.5 * (lower + upper)
    horizontal_extent = float(max(upper[0] - lower[0], upper[1] - lower[1]))
    eye = center + horizontal_extent * np.asarray((0.92, -1.05, 0.80))
    forward = center - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray((0.0, 0.0, 1.0))
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    fov_rad = np.deg2rad(42.0)
    tangent = float(np.tan(0.5 * fov_rad))
    x = (2.0 * (np.arange(size, dtype=np.float64) + 0.5) / size - 1.0) * tangent
    y = (1.0 - 2.0 * (np.arange(size, dtype=np.float64) + 0.5) / size) * tangent
    grid_x, grid_y = np.meshgrid(x, y)
    directions = (
        forward[None, None]
        + grid_x[..., None] * right[None, None]
        + grid_y[..., None] * up[None, None]
    )
    origins = np.broadcast_to(eye, directions.shape)
    camera = {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": up,
        "tangent": tangent,
        "size": float(size),
    }
    return origins.astype(np.float32), directions.astype(np.float32), camera


def project_overview(point: np.ndarray, camera: dict[str, np.ndarray | float]) -> np.ndarray:
    relative = point - np.asarray(camera["eye"])
    z = float(relative @ np.asarray(camera["forward"]))
    x = float(relative @ np.asarray(camera["right"]))
    y = float(relative @ np.asarray(camera["up"]))
    tangent = float(camera["tangent"])
    size = float(camera["size"])
    return np.asarray(
        (
            0.5 * size * (1.0 + x / (z * tangent)),
            0.5 * size * (1.0 - y / (z * tangent)),
        )
    )


def prepared_prior_render(sample: np.lib.npyio.NpzFile) -> np.ndarray:
    valid = sample["bim_valid"].astype(bool)
    category = sample["bim_category"].astype(np.int64)
    normals = sample["bim_normals"].astype(np.float32).transpose(1, 2, 0)
    image = np.broadcast_to(BACKGROUND, (*valid.shape, 3)).copy()
    usable = valid & (category >= 0) & (category < len(ENVELOPE_CATEGORIES))
    light = np.asarray((-0.35, -0.45, 0.82), dtype=np.float32)
    light /= np.linalg.norm(light)
    illumination = 0.58 + 0.42 * np.abs(np.sum(normals * light, axis=-1))
    image[usable] = CLASS_COLORS[category[usable]] * illumination[usable, None]
    image[cv2.Canny(valid.astype(np.uint8) * 255, 20, 80) > 0] = 0.12
    return np.clip(image, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    record = find_record(args.manifest, args.sample_id)
    room = args.sample_id.split("/", 1)[0]
    ifc_path = args.ifc_dir / f"{room}.ifc"
    vertices_local, triangles, face_classes, object_counts = load_full_ifc(
        ifc_path,
        args.workers,
    )

    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    area_from_bim = np.asarray(
        alignment["rooms"][room]["T_area_from_bim"],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        (vertices_local.astype(np.float64), np.ones((len(vertices_local), 1))),
        axis=1,
    )
    vertices_area = (homogeneous @ area_from_bim.T)[:, :3].astype(np.float32)

    pose = json.loads(Path(str(record["pose"])).read_text(encoding="utf-8"))
    intrinsic_source, _, camera_to_area = pose_matrices(pose)
    image_bgr = cv2.imread(str(record["image"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {record['image']}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
    scale_x = args.size / image_bgr.shape[1]
    scale_y = args.size / image_bgr.shape[0]
    intrinsic = intrinsic_source.copy()
    intrinsic[0] *= scale_x
    intrinsic[1] *= scale_y

    scene_area = make_scene(vertices_area, triangles)
    frame_origins, frame_directions = camera_rays(
        intrinsic,
        camera_to_area,
        args.size,
        args.size,
    )
    frame_render, _ = category_render(
        scene_area,
        face_classes,
        frame_origins,
        frame_directions,
    )

    # Hide only the ceiling in the overview so the room contents remain visible;
    # the exact frame render below still contains the untouched full IFC.
    overview_mask = face_classes != ENVELOPE_CATEGORIES.index("ceiling")
    scene_local = make_scene(vertices_local, triangles[overview_mask])
    overview_origins, overview_directions, overview_camera = overview_rays(
        vertices_local,
        args.size,
    )
    overview, _ = category_render(
        scene_local,
        face_classes[overview_mask],
        overview_origins,
        overview_directions,
    )

    bim_from_area = np.linalg.inv(area_from_bim)
    camera_local_h = bim_from_area @ np.append(camera_to_area[:3, 3], 1.0)
    camera_forward_area = camera_to_area[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
    camera_forward_local = bim_from_area[:3, :3] @ camera_forward_area
    camera_forward_local /= np.linalg.norm(camera_forward_local)
    marker = project_overview(camera_local_h[:3], overview_camera)
    marker_forward = project_overview(
        camera_local_h[:3] + 1.25 * camera_forward_local,
        overview_camera,
    )

    sample = np.load(str(record["sample"]))
    prior = prepared_prior_render(sample)
    coverage = float(sample["bim_valid"].astype(bool).mean())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB regular view")
    axes[1].imshow(overview)
    axes[1].annotate(
        "",
        xy=marker_forward,
        xytext=marker,
        arrowprops={"arrowstyle": "-|>", "color": "#00e5ff", "lw": 2.2},
    )
    axes[1].scatter(marker[0], marker[1], s=45, c="#00e5ff", edgecolors="black")
    axes[1].set_title("Original office_31 IFC cutaway\nceiling hidden; cyan = frame_23 camera")
    axes[2].imshow(frame_render)
    axes[2].set_title("Original full IFC\nrendered from frame_23 pose")
    axes[3].imshow(prior)
    axes[3].set_title(f"Filtered structural BIM actually used\nvalid coverage = {coverage:.1%}")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    shown_classes = [index for index, name in enumerate(CLASS_NAMES) if object_counts.get(name, 0)]
    labels = {
        "wall": "wall",
        "floor": "floor",
        "ceiling": "ceiling",
        "column": "column",
        "beam": "beam",
        "furniture": "furniture",
        "proxy": "proxy",
        "other": "other",
    }
    handles = [
        Patch(
            facecolor=CLASS_COLORS[index],
            edgecolor="0.2",
            label=f"{labels.get(CLASS_NAMES[index], CLASS_NAMES[index])} "
            f"({object_counts.get(CLASS_NAMES[index], 0)} objects)",
        )
        for index in shown_classes
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(7, len(handles)),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    fig.suptitle(
        f"{args.sample_id} | original BIMSyn IFC versus deployed structural prior",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
