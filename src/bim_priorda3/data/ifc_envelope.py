from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

ENVELOPE_CATEGORIES = (
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "column",
    "beam",
)
GLOBAL_CORE_CATEGORIES = ("wall", "floor", "ceiling", "column", "beam")


def _predefined_type(product: Any) -> str:
    value = getattr(product, "PredefinedType", None)
    return "" if value is None else str(value).upper()


def envelope_category(product: Any) -> str | None:
    """Return the fixed-envelope category for an IFC product.

    The function is deliberately an allow-list.  BIMSyn's IFC files also contain
    furniture, boards, clutter-like proxies, openings and MEP terminals; accepting
    generic ``IfcBuildingElement`` or ``IfcBuildingElementProxy`` instances would
    leak the foreground geometry that the model is meant to predict.
    """

    if product.is_a("IfcWall") or product.is_a("IfcCurtainWall"):
        return "wall"
    if product.is_a("IfcSlab"):
        slab_type = _predefined_type(product)
        if slab_type in {"ROOF"}:
            return "ceiling"
        if slab_type in {"", "NOTDEFINED", "USERDEFINED", "FLOOR", "BASESLAB"}:
            return "floor"
        return None
    if product.is_a("IfcCovering"):
        covering_type = _predefined_type(product)
        if covering_type == "CEILING":
            return "ceiling"
        if covering_type == "FLOORING":
            return "floor"
        return None
    if product.is_a("IfcRoof"):
        return "ceiling"
    if product.is_a("IfcDoor"):
        return "door"
    if product.is_a("IfcWindow"):
        return "window"
    if product.is_a("IfcColumn"):
        return "column"
    if product.is_a("IfcBeam"):
        return "beam"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IFCEnvelopeGeometry:
    """Triangulated, room-local envelope geometry in metres."""

    vertices: np.ndarray
    triangles: np.ndarray
    triangle_categories: np.ndarray
    category_names: tuple[str, ...]
    audit: dict[str, Any]

    def tensor_mesh(self) -> o3d.t.geometry.TriangleMesh:
        return o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(
                np.ascontiguousarray(self.vertices),
                dtype=o3d.core.Dtype.Float32,
            ),
            o3d.core.Tensor(
                np.ascontiguousarray(self.triangles),
                dtype=o3d.core.Dtype.Int32,
            ),
        )

    def category_triangles(self, categories: set[str]) -> np.ndarray:
        wanted = {index for index, name in enumerate(self.category_names) if name in categories}
        if not wanted:
            return np.empty((0, 3), dtype=np.int32)
        mask = np.isin(self.triangle_categories, np.fromiter(wanted, dtype=np.uint8))
        return self.triangles[mask]


def load_ifc_envelope_geometry(
    path: str | Path,
    *,
    strict: bool = True,
) -> IFCEnvelopeGeometry:
    """Triangulate only the fixed building envelope from one BIMSyn IFC file.

    IfcOpenShell applies every product's ``ObjectPlacement`` and emits metre-based
    coordinates.  The returned coordinate system remains the room-local BIM frame;
    registration to the Stanford Area frame is handled once per room elsewhere.
    """

    try:
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.unit
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "IFC envelope preparation requires the optional 'ifcopenshell' package"
        ) from error

    ifc_path = Path(path).expanduser().resolve()
    if not ifc_path.is_file():
        raise FileNotFoundError(f"IFC file does not exist: {ifc_path}")
    model = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("convert-back-units", False)
    settings.set("weld-vertices", True)

    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    triangle_categories: list[np.ndarray] = []
    represented_counts: Counter[str] = Counter()
    included_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    triangle_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    products = sorted(model.by_type("IfcProduct"), key=lambda item: int(item.id()))
    for product in products:
        if getattr(product, "Representation", None) is None:
            continue
        ifc_class = str(product.is_a())
        represented_counts[ifc_class] += 1
        category = envelope_category(product)
        if category is None:
            excluded_counts[ifc_class] += 1
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            product_vertices = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
            product_triangles = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
            if not len(product_vertices) or not len(product_triangles):
                raise ValueError("triangulation returned an empty mesh")
        except Exception as error:  # IfcOpenShell has multiple backend exception types.
            failure = {
                "step_id": int(product.id()),
                "global_id": str(getattr(product, "GlobalId", "")),
                "ifc_class": ifc_class,
                "name": str(getattr(product, "Name", "") or ""),
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            if strict:
                raise RuntimeError(
                    f"Failed to triangulate selected envelope element "
                    f"#{product.id()} in {ifc_path}: {error}"
                ) from error
            continue

        offset = sum(len(item) for item in vertices)
        category_index = ENVELOPE_CATEGORIES.index(category)
        vertices.append(product_vertices.astype(np.float32))
        triangles.append((product_triangles + offset).astype(np.int32))
        triangle_categories.append(np.full(len(product_triangles), category_index, dtype=np.uint8))
        included_counts[category] += 1
        triangle_counts[category] += len(product_triangles)

    if not triangles:
        raise RuntimeError(f"No fixed-envelope geometry was selected from {ifc_path}")

    merged_vertices = np.concatenate(vertices)
    merged_triangles = np.concatenate(triangles)
    merged_categories = np.concatenate(triangle_categories)
    unit_scale = float(ifcopenshell.util.unit.calculate_unit_scale(model))
    audit: dict[str, Any] = {
        "schema_version": 1,
        "source_ifc": str(ifc_path),
        "source_sha256": _sha256(ifc_path),
        "ifc_schema": str(model.schema),
        "declared_length_unit_to_m": unit_scale,
        "geometry_output_unit": "metre",
        "coordinate_frame": "room-local BIM",
        "filter_policy": "fixed-envelope-allowlist-v1",
        "allowed_categories": list(ENVELOPE_CATEGORIES),
        "represented_ifc_class_counts": dict(sorted(represented_counts.items())),
        "included_element_counts": dict(sorted(included_counts.items())),
        "excluded_ifc_class_counts": dict(sorted(excluded_counts.items())),
        "triangle_counts": dict(sorted(triangle_counts.items())),
        "vertices": len(merged_vertices),
        "triangles": len(merged_triangles),
        "bounds_min_m": merged_vertices.min(axis=0).astype(float).tolist(),
        "bounds_max_m": merged_vertices.max(axis=0).astype(float).tolist(),
        "triangulation_failures": failures,
    }
    return IFCEnvelopeGeometry(
        vertices=merged_vertices,
        triangles=merged_triangles,
        triangle_categories=merged_categories,
        category_names=ENVELOPE_CATEGORIES,
        audit=audit,
    )


def build_ifc_envelope_scene(
    path: str | Path,
    *,
    strict: bool = True,
) -> tuple[o3d.t.geometry.RaycastingScene, IFCEnvelopeGeometry]:
    geometry = load_ifc_envelope_geometry(path, strict=strict)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(geometry.tensor_mesh())
    return scene, geometry


def build_global_ifc_envelope_scene(
    ifc_paths: dict[str, Path],
    area_from_bim: dict[str, np.ndarray],
    *,
    included_categories: tuple[str, ...] = GLOBAL_CORE_CATEGORIES,
    strict: bool = True,
) -> tuple[o3d.t.geometry.RaycastingScene, IFCEnvelopeGeometry]:
    """Merge registered room IFCs into one fixed Area-coordinate prior.

    Door and window leaves are deliberately absent from the default core protocol:
    their BIM closed/open state is not synchronized with the captured imagery and
    can otherwise occlude rays through doorways.  All transforms are the fixed,
    per-room registration outputs; no frame depth or RGB enters this operation.
    """

    rooms = sorted(ifc_paths)
    if not rooms:
        raise ValueError("Global IFC scene requires at least one room")
    if set(rooms) != set(area_from_bim):
        missing_transforms = sorted(set(rooms) - set(area_from_bim))
        missing_ifcs = sorted(set(area_from_bim) - set(rooms))
        raise ValueError(
            "IFC/transform room sets differ: "
            f"missing transforms={missing_transforms}, missing IFCs={missing_ifcs}"
        )
    included = tuple(dict.fromkeys(included_categories))
    unknown = sorted(set(included) - set(ENVELOPE_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown global envelope categories: {unknown}")
    if not included:
        raise ValueError("Global envelope category allow-list must not be empty")

    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    triangle_categories: list[np.ndarray] = []
    room_audits: dict[str, Any] = {}
    included_triangle_counts: Counter[str] = Counter()
    excluded_triangle_counts: Counter[str] = Counter()
    offset = 0
    for room in rooms:
        transform = np.asarray(area_from_bim[room], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"{room}: T_area_from_bim must be a finite 4x4 matrix")
        geometry = load_ifc_envelope_geometry(ifc_paths[room], strict=strict)
        mask = np.isin(
            geometry.triangle_categories,
            np.asarray(
                [ENVELOPE_CATEGORIES.index(category) for category in included],
                dtype=np.uint8,
            ),
        )
        selected_triangles = geometry.triangles[mask]
        selected_categories = geometry.triangle_categories[mask]
        for category_index, category in enumerate(ENVELOPE_CATEGORIES):
            category_mask = geometry.triangle_categories == category_index
            included_triangle_counts[category] += int((category_mask & mask).sum())
            excluded_triangle_counts[category] += int((category_mask & ~mask).sum())
        if not len(selected_triangles):
            raise RuntimeError(f"{room}: no triangles survived the global core filter")

        homogeneous = np.concatenate(
            (
                geometry.vertices.astype(np.float64),
                np.ones((len(geometry.vertices), 1), dtype=np.float64),
            ),
            axis=1,
        )
        area_vertices = (homogeneous @ transform.T)[:, :3].astype(np.float32)
        vertices.append(area_vertices)
        triangles.append((selected_triangles + offset).astype(np.int32))
        triangle_categories.append(selected_categories.astype(np.uint8))
        offset += len(area_vertices)
        room_audits[room] = {
            "source_ifc_sha256": geometry.audit["source_sha256"],
            "T_area_from_bim": transform.astype(float).tolist(),
            "source_geometry": geometry.audit,
            "selected_triangles": int(mask.sum()),
            "excluded_triangles": int((~mask).sum()),
        }

    merged_vertices = np.concatenate(vertices)
    merged_triangles = np.concatenate(triangles)
    merged_categories = np.concatenate(triangle_categories)
    if included == GLOBAL_CORE_CATEGORIES:
        filter_policy = "global-area-fixed-core-envelope-v1"
        dynamic_state_rationale = "door/window omitted: capture state is unsynchronized"
    elif included == ENVELOPE_CATEGORIES:
        filter_policy = "global-area-fixed-envelope-v2"
        dynamic_state_rationale = "door/window retained; non-envelope foreground remains excluded"
    else:
        filter_policy = "global-area-custom-envelope-v1"
        dynamic_state_rationale = "explicit caller-provided envelope allow-list"
    audit: dict[str, Any] = {
        "schema_version": 1,
        "coordinate_frame": "Stanford Area_1 world, metres, Z-up",
        "filter_policy": filter_policy,
        "included_categories": list(included),
        "excluded_envelope_categories": sorted(set(ENVELOPE_CATEGORIES) - set(included)),
        "dynamic_state_rationale": dynamic_state_rationale,
        "rooms": rooms,
        "room_sources": room_audits,
        "included_triangle_counts": dict(sorted(included_triangle_counts.items())),
        "excluded_triangle_counts": dict(sorted(excluded_triangle_counts.items())),
        "vertices": len(merged_vertices),
        "triangles": len(merged_triangles),
        "bounds_min_m": merged_vertices.min(axis=0).astype(float).tolist(),
        "bounds_max_m": merged_vertices.max(axis=0).astype(float).tolist(),
    }
    geometry = IFCEnvelopeGeometry(
        vertices=merged_vertices,
        triangles=merged_triangles,
        triangle_categories=merged_categories,
        category_names=ENVELOPE_CATEGORIES,
        audit=audit,
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(geometry.tensor_mesh())
    return scene, geometry


def render_ifc_envelope(
    scene: o3d.t.geometry.RaycastingScene,
    geometry: IFCEnvelopeGeometry,
    intrinsic: np.ndarray,
    camera_to_scene: np.ndarray,
    height: int,
    width: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render z-depth, camera-space normals and envelope category IDs."""

    if intrinsic.shape != (3, 3) or camera_to_scene.shape != (4, 4):
        raise ValueError("Expected a 3x3 intrinsic and 4x4 camera-to-scene transform")
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    directions_camera = pixels @ np.linalg.inv(intrinsic).T
    directions_scene = directions_camera @ camera_to_scene[:3, :3].T
    origins = np.broadcast_to(camera_to_scene[:3, 3], directions_scene.shape)
    rays = np.concatenate((origins, directions_scene), axis=-1).astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))

    depth = result["t_hit"].numpy().astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= float(max_depth))
    depth[~valid] = np.nan

    normals_scene = result["primitive_normals"].numpy().astype(np.float32)
    normals_camera = normals_scene @ camera_to_scene[:3, :3]
    normals_camera[~valid] = 0.0
    normals = normals_camera.transpose(2, 0, 1).astype(np.float32)

    primitive_ids = result["primitive_ids"].numpy().astype(np.int64)
    category = np.full((height, width), 255, dtype=np.uint8)
    hit = valid & (primitive_ids >= 0) & (primitive_ids < len(geometry.triangles))
    category[hit] = geometry.triangle_categories[primitive_ids[hit]]
    return depth, normals, category
